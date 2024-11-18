# -*- coding: utf-8 -*-
"""
모델 학습을 위한 스크립트 (로컬 환경).
"""

import os
import subprocess
import numpy as np
import faiss
from pathlib import Path
import json
from random import shuffle
import sys
import shutil
import tempfile

# OpenMP 충돌 문제 해결
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

# RMVPE 모델 경로 설정
os.environ["RMVPE_MODEL_PATH"] = str(Path("C:/Users/한윤택.HYT/Desktop/last/KiwKiw/assets/rmvpe/rmvpe.pt"))

# 현재 디렉토리 설정
now_dir = Path("C:/Users/한윤택.HYT/Desktop/last/KiwKiw")
sys.path.append(str(now_dir))

# 데이터 전처리
def preprocess_data(dataset, sample_rate, exp_dir, now_dir):
    command = [
        "python",
        str(now_dir / "infer/modules/train/preprocess.py"),
        dataset,
        str(sample_rate),
        "2",  # Number of processes
        exp_dir,
        "False",
        "3.0",  # Default `per` value
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(now_dir)
    print(f"Running preprocess: {' '.join(command)}")
    subprocess.run(command, check=True, env=env)

# Feature 추출
def extract_features(exp_dir, now_dir, version, f0_method):
    command = [
        "python",
        str(now_dir / "infer/modules/train/extract/extract_f0_rmvpe.py"),
        "1",
        "0",
        "0",
        str(now_dir / "logs" / exp_dir),
        "True"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(now_dir)
    print(f"Extracting f0 with command: {' '.join(command)}")
    subprocess.run(command, check=True, env=env)

# FAISS Index 훈련
def train_index(exp_dir, version, now_dir):
    exp_dir_path = now_dir / "logs" / exp_dir
    feature_dir = exp_dir_path / "2b-f0nsf"

    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")

    npy_files = sorted(feature_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} feature files.")

    # Load and concatenate numpy files
    npys = [np.load(file) for file in npy_files]
    big_npy = np.concatenate(npys, axis=0)
    if big_npy.ndim == 1:
        big_npy = big_npy.reshape(-1, 1)

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    d = big_npy.shape[1]
    index = faiss.index_factory(d, f"IVF{n_ivf},Flat")
    index.train(big_npy)

        # FAISS index 저장
    # 강제로 임시 파일 경로를 영어 경로로 지정
    temp_dir = "C:/Temp"
    os.makedirs(temp_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".index", dir=temp_dir)
    temp_path = temp_file.name
    temp_file.close()

    print(f"Temporary FAISS index path: {temp_path}")
    faiss.write_index(index, temp_path)

    # 최종 저장 경로
    output_path = exp_dir_path / f"trained_IVF{n_ivf}_Flat.index"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 기존 파일 삭제
    if output_path.exists():
        print(f"File already exists at {output_path}, removing it...")
        output_path.unlink()

    # 임시 파일 이동
    try:
        shutil.move(temp_path, output_path)
        print(f"Saved FAISS index to: {output_path}")
    except Exception as e:
        print(f"Failed to move FAISS index: {e}")
        os.remove(temp_path)
        raise

# 모델 학습
def train_model(
    exp_dir, sr, f0_method, spk_id, save_freq, total_epoch,
    batch_size, gpus, pretrained_G, pretrained_D, cache_gpu, version, now_dir
):
    filelist_path = now_dir / "logs" / exp_dir / "filelist.txt"
    config_path = now_dir / "configs" / version / "48k.json"
    cmd = [
        "python",
        str(now_dir / "infer/modules/train/train.py"),
        "-e", exp_dir,
        "-sr", str(sr),
        "-f0", str(f0_method),
        "-bs", str(batch_size),
        "-g", str(gpus),
        "-te", str(total_epoch),
        "-se", str(save_freq),
        "-pg", pretrained_G,
        "-pd", pretrained_D,
        "-l", "1",
        "-c", "1",
        "-sw", "1",
        "-v", version,
    ]
    print(f"Starting training with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# 결과 ZIP
def zip_and_save_results(exp_dir, now_dir):
    zip_file = now_dir / "logs" / exp_dir / f"{exp_dir}.zip"
    result_dir = now_dir / "logs" / exp_dir
    print(f"Zipping results into {zip_file}...")
    subprocess.run(["zip", "-r", str(zip_file), str(result_dir)], check=True)

# 메인 실행 흐름
def main():
    exp_dir = "sample_model"
    dataset = "processed_audio/IU_Live/htdemucs/IU_Live"
    sample_rate = 48000
    version = "v2"
    f0_method = 1
    save_freq = 50
    total_epoch = 10
    batch_size = 7
    cache_gpu = True
    spk_id = 0
    pretrained_G = str(now_dir / "assets/pretrained_v2/f0G48k.pth")
    pretrained_D = str(now_dir / "assets/pretrained_v2/f0D48k.pth")

    preprocess_data(dataset, sample_rate, exp_dir, now_dir)
    extract_features(exp_dir, now_dir, version, f0_method)
    train_index(exp_dir, version, now_dir)
    train_model(exp_dir, sample_rate, f0_method, spk_id, save_freq, total_epoch, batch_size, 0, pretrained_G, pretrained_D, cache_gpu, version, now_dir)
    zip_and_save_results(exp_dir, now_dir)

if __name__ == "__main__":
    main()
