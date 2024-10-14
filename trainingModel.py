# -*- coding: utf-8 -*-
"""
모델 학습을 위한 스크립트 (Google Drive 저장 없이 로컬 저장으로 전환).
"""

import os
import subprocess
import numpy as np
import faiss
from subprocess import Popen, PIPE, STDOUT
import pathlib
import json
from random import shuffle

# 설치할 패키지 목록
def install_packages():
    """
    필요한 패키지들을 설치합니다.
    """
    print("Installing necessary packages...")
    subprocess.run(['pip', 'install', 'pip==23.3.1'])
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
    subprocess.run(['pip', 'install', 'ffmpeg', 'av', 'ffmpeg-python', 'praat-parselmouth', 'pyworld'])
    subprocess.run(['pip3', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])
    subprocess.run(['pip', 'install', 'fairseq'])

# 데이터 전처리
def preprocess_data(dataset, sample_rate, exp_dir, now_dir):
    """
    데이터 전처리를 수행하는 함수.
    """
    command = f"python infer/modules/train/preprocess.py '{dataset}' {sample_rate} 2 '{now_dir}/logs/{exp_dir}' False 3.0"
    print(f"Running preprocess: {command}")
    subprocess.run(command, shell=True, check=True)

# Feature 추출
def extract_features(exp_dir, now_dir, version, f0method):
    """
    음성 특성(f0, feature) 추출을 위한 함수.
    """
    if f0method != "rmvpe_gpu":
        command = f"python infer/modules/train/extract/extract_f0_print.py '{now_dir}/logs/{exp_dir}' 2 '{f0method}'"
    else:
        command = f"python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 '{now_dir}/logs/{exp_dir}' True"
    print(f"Extracting f0: {command}")
    subprocess.run(command, shell=True, check=True)

    command = f"python infer/modules/train/extract_feature_print.py cuda:0 1 0 0 '{now_dir}/logs/{exp_dir}' '{version}' False"
    print(f"Extracting feature: {command}")
    subprocess.run(command, shell=True, check=True)

# Feature Index 훈련
def train_index(exp_dir, version19, now_dir):
    """
    Feature Index를 훈련하는 함수.
    """
    exp_dir_path = f"{now_dir}/logs/{exp_dir}"
    feature_dir = f"{exp_dir_path}/3_feature768" if version19 == "v2" else f"{exp_dir_path}/3_feature256"
    
    if not os.path.exists(feature_dir):
        return "Feature extraction must be done first!"

    npys = [np.load(f"{feature_dir}/{name}") for name in sorted(os.listdir(feature_dir))]
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(256 if version19 == "v1" else 768, f"IVF{n_ivf},Flat")
    index.train(big_npy)
    faiss.write_index(index, f"{exp_dir_path}/trained_IVF{n_ivf}_Flat.index")

    print("Feature index training complete.")

# 모델 학습 함수
def train_model(
    exp_dir,
    sr,
    f0_method,
    spk_id,
    save_freq,
    total_epoch,
    batch_size,
    gpus,
    pretrained_G,
    pretrained_D,
    cache_gpu,
    version,
    now_dir
):
    """
    학습을 수행하는 메인 함수.
    """
    # 학습을 위한 filelist 생성
    gt_wavs_dir = f"{now_dir}/logs/{exp_dir}/0_gt_wavs"
    feature_dir = f"{now_dir}/logs/{exp_dir}/3_feature768" if version == "v2" else f"{now_dir}/logs/{exp_dir}/3_feature256"
    
    names = set(os.listdir(gt_wavs_dir)) & set(os.listdir(feature_dir))
    filelist = [
        f"{gt_wavs_dir}/{name}|{feature_dir}/{name}.npy|{spk_id}" for name in names
    ]

    shuffle(filelist)
    with open(f"{now_dir}/logs/{exp_dir}/filelist.txt", "w") as f:
        f.write("\n".join(filelist))

    # config 파일 설정
    config_path = f"configs/v2/{sr}.json" if version == "v2" else f"configs/v1/{sr}.json"
    config_save_path = f"{now_dir}/logs/{exp_dir}/config.json"
    if not pathlib.Path(config_save_path).exists():
        with open(config_path, "r") as config_file:
            config_data = json.load(config_file)
            with open(config_save_path, "w") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=4, sort_keys=True)

    # 학습 실행
    cmd = (
        f'python infer/modules/train/train.py -e "{exp_dir}" -sr {sr} -f0 {f0_method} -bs {batch_size} '
        f'-g {gpus} -te {total_epoch} -se {save_freq} {"-pg " + pretrained_G if pretrained_G else ""} '
        f'{"-pd " + pretrained_D if pretrained_D else ""} -l {cache_gpu} -c {cache_gpu} -sw {cache_gpu} -v {version}'
    )
    print(f"Starting model training: {cmd}")
    p = Popen(cmd, shell=True, cwd=now_dir, stdout=PIPE, stderr=STDOUT, bufsize=1, universal_newlines=True)
    for line in p.stdout:
        print(line.strip())
    p.wait()

# 압축 및 결과 저장 (로컬)
def zip_and_save_results(exp_dir, now_dir):
    """
    학습 결과를 압축하여 로컬에 저장합니다.
    """
    zip_file = f"{now_dir}/logs/{exp_dir}/{exp_dir}.zip"
    result_files = [
        f"{now_dir}/logs/{exp_dir}/added_*.index",
        f"{now_dir}/logs/{exp_dir}/total_*.npy",
        f"{now_dir}/assets/weights/{exp_dir}.pth"
    ]
    
    print(f"Zipping results into {zip_file}...")
    subprocess.run(['zip', '-r', zip_file] + result_files, check=True)

# 전체 실행 흐름
def main():
    now_dir = "/content/KiwKiw"
    exp_dir = "sample_model"
    dataset = "/path/to/your/dataset"
    sample_rate = "48000"
    version = "v2"
    f0_method = "rmvpe_gpu"
    save_freq = 50
    total_epoch = 10
    batch_size = "7"
    cache_gpu = True
    spk_id = 0
    pretrained_G = "assets/pretrained_v2/f0G48k.pth"
    pretrained_D = "assets/pretrained_v2/f0D48k.pth"
    
    install_packages()
    preprocess_data(dataset, sample_rate, exp_dir, now_dir)
    extract_features(exp_dir, now_dir, version, f0_method)
    train_index(exp_dir, version, now_dir)
    train_model(exp_dir, sample_rate, 1, spk_id, save_freq, total_epoch, batch_size, 0, pretrained_G, pretrained_D, cache_gpu, version, now_dir)
    zip_and_save_results(exp_dir, now_dir)

# 실행
if __name__ == "__main__":
    main()
