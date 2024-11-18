import os
import subprocess
import json
from pathlib import Path
from random import shuffle
from subprocess import Popen, PIPE, STDOUT
import sys
import numpy as np


# 로컬 환경 설정
model_name = "TestCase"  # 모델 이름
exp_dir = model_name
dataset = "splitted/TestCase"  # 로컬 데이터셋 경로
sample_rate = "48000"  # 샘플링 레이트 설정
ksample_rate = "48k" if sample_rate == "48000" else "40k"
version = "v2"  # 버전 설정 ["v1", "v2"]
f0method = "rmvpe_gpu"  # F0 추출 방법 ["pm", "dio", "harvest", "rmvpe", "rmvpe_gpu"]

save_frequency = 50  # 모델 저장 빈도
epoch = 10  # 학습 에포크 수
batch_size = "7"  # 배치 크기
cache_gpu = True  # GPU 캐시 사용 여부

# 현재 디렉토리 설정
now_dir = "KiwKiw"

# 경로 생성 및 로그 파일 초기화
os.makedirs(f"{now_dir}/logs/{exp_dir}", exist_ok=True)
with open(f"{now_dir}/logs/{exp_dir}/preprocess.log", "w") as f:
    pass
with open(f"{now_dir}/logs/{exp_dir}/extract_f0_feature.log", "w") as f:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__))
kiwkiw_root = os.path.join(current_dir, "KiwKiw")
if kiwkiw_root not in sys.path:
    sys.path.append(kiwkiw_root)

print(f"Model directory prepared: {now_dir}/logs/{exp_dir}")

def create_config_file():
    config_path = "KiwKiw/configs/v2/48k.json"  # 템플릿 경로
    config_save_path = "KiwKiw/logs/TestCase/config.json"  # 저장 경로

    # 템플릿 파일이 존재하는지 확인
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Template config file not found: {config_path}")

    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as template_file:
        config_data = json.load(template_file)

    with open(config_save_path, "w", encoding="utf-8") as config_file:
        json.dump(config_data, config_file, indent=4)
    print(f"Config file created: {config_save_path}")


# 데이터 전처리
def preprocess_data():
    command = [
        "python",
        f"{now_dir}/infer/modules/train/preprocess.py",
        dataset,
        sample_rate,
        "2",  # 병렬 처리 수
        f"{now_dir}/logs/{exp_dir}",
        "False",
        "3.0",  # Default value for 'per'
    ]
    print(f"Running preprocess command: {' '.join(command)}")
    subprocess.run(command, check=True)

# F0 특성 추출
def extract_f0_features():
    command = [
        "python",
        "KiwKiw/infer/modules/train/extract/extract_f0_rmvpe.py",
        "1",
        "0",
        "0",
        "KiwKiw/logs/TestCase",
        "True"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("KiwKiw")
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    print(f"Running extract_f0_features command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command, 
            check=True, 
            env=env, 
            capture_output=True, 
            text=True,  # 텍스트 모드로 실행
            encoding="utf-8"  # 출력 디코딩을 UTF-8로 강제
        )
        print("Command output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred during command execution:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise


def mock_feature_creation(exp_dir):
    feature_dir = os.path.join(exp_dir, "3_feature768")
    os.makedirs(feature_dir, exist_ok=True)
    # 더미 데이터 생성
    for i in range(5):  # 임의로 5개의 더미 파일 생성
        dummy_data = np.random.rand(768)
        np.save(os.path.join(feature_dir, f"dummy_feature_{i}.npy"), dummy_data)

# FAISS Feature Index 학습
def train_feature_index():

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import numpy as np
    import faiss

    feature_dir = (
        f"{now_dir}/logs/{exp_dir}/3_feature256" if version == "v1" else f"{now_dir}/logs/{exp_dir}/3_feature768"
    )

    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    print(f"Found feature directory: {feature_dir}")

    npys = [np.load(f"{feature_dir}/{file}") for file in os.listdir(feature_dir)]
    big_npy = np.concatenate(npys, axis=0)
    print(f"Concatenated big_npy shape: {big_npy.shape}")
    # pipline2.py 수정
    if big_npy.ndim == 1:
        print(f"Reshaping big_npy with shape {big_npy.shape} to 2D array")
        big_npy = big_npy.reshape(-1, 1)

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(big_npy.shape[1], f"IVF{n_ivf},Flat")
    print(f"Training FAISS index with IVF{n_ivf}...")

    index.train(big_npy)
    index_path = f"{now_dir}/logs/{exp_dir}/trained_IVF{n_ivf}_Flat.index"
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to: {index_path}")

# 모델 학습
def train_model():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    gt_wavs_dir = f"{now_dir}/logs/{exp_dir}/0_gt_wavs"
    feature_dir = f"{now_dir}/logs/{exp_dir}/3_feature256" if version == "v1" else f"{now_dir}/logs/{exp_dir}/3_feature768"
    pitch_dir = f"{now_dir}/logs/{exp_dir}/2a_f0"
    pitchf_dir = f"{now_dir}/logs/{exp_dir}/2b-f0nsf"
    filelist_path = f"{now_dir}/logs/{exp_dir}/filelist.txt"

    # 디렉토리 확인
    if not os.path.exists(gt_wavs_dir):
        raise FileNotFoundError(f"GT WAV directory not found: {gt_wavs_dir}")
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    if not os.path.exists(pitch_dir):
        raise FileNotFoundError(f"Pitch directory not found: {pitch_dir}")
    if not os.path.exists(pitchf_dir):
        raise FileNotFoundError(f"PitchF directory not found: {pitchf_dir}")

    # 파일 이름 읽기
    gt_wavs_files = sorted(os.listdir(gt_wavs_dir))
    feature_files = sorted(os.listdir(feature_dir))
    pitch_files = sorted(os.listdir(pitch_dir))
    pitchf_files = sorted(os.listdir(pitchf_dir))

    # 이름에서 확장자 제거
    gt_wavs_names = [f.split(".")[0] for f in gt_wavs_files if f.endswith(".wav")]
    feature_names = [f.split(".")[0] for f in feature_files if f.endswith(".npy")]
    pitch_names = [f.split(".")[0] for f in pitch_files if f.endswith(".npy")]
    pitchf_names = [f.split(".")[0] for f in pitchf_files if f.endswith(".npy")]

    # 매핑에 따라 filelist 생성
    filelist = [
        f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{pitch_dir}/{name}.npy|{pitchf_dir}/{name}.npy|0"
        for name in gt_wavs_names
        if name in feature_names and name in pitch_names and name in pitchf_names
    ]

    # 매핑되지 않은 파일 처리
    unmapped_files = set(gt_wavs_names) - set(feature_names) - set(pitch_names) - set(pitchf_names)
    if unmapped_files:
        print(f"Warning: Some files could not be mapped: {unmapped_files}")

    # Shuffle and save filelist
    shuffle(filelist)
    with open(filelist_path, "w") as f:
        f.write("\n".join(filelist))
    print(f"Filelist saved to: {filelist_path}")

    # Config 준비
    config_path = f"{now_dir}/configs/v2/48k.json"
    config_save_path = f"{now_dir}/logs/{exp_dir}/config.json"
    if not os.path.exists(config_save_path):
        with open(config_path, "r") as src, open(config_save_path, "w") as dst:
            json.dump(json.load(src), dst, indent=4)
    print(f"Config file saved to: {config_save_path}")

    # Training 명령 실행
    cmd = [
        "python",
        f"{now_dir}/infer/modules/train/train.py",
        "-e", exp_dir,
        "-sr", sample_rate,
        "-f0", "1",
        "-bs", batch_size,
        "-g", "0",
        "-te", str(epoch),
        "-se", str(save_frequency),
        "-pg", f"{now_dir}/assets/pretrained_v2/f0G48k.pth",
        "-pd", f"{now_dir}/assets/pretrained_v2/f0D48k.pth",
        "-l", "1",
        "-c", "1",
        "-sw", "1",
        "-v", version,
    ]
    print(f"Starting training with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# 실행 흐름
def main():
    preprocess_data()
    extract_f0_features()
    mock_feature_creation("KiwKiw/logs/TestCase")
    train_feature_index()
    train_model()

if __name__ == "__main__":
    main()