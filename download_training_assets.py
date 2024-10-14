# -*- coding: utf-8 -*-
"""
훈련을 위한 미리 학습된 모델 파일을 다운로드하고, 로컬 파일 시스템에 저장하는 스크립트.
"""

# 필수 라이브러리 임포트
import os
import subprocess

# 다운로드를 위한 aria2 설치 함수
def install_aria2():
    """
    aria2 설치 여부 확인 후 설치합니다.
    """
    try:
        subprocess.run(['aria2c', '--version'], check=True)
        print("aria2 already installed.")
    except subprocess.CalledProcessError:
        print("Installing aria2...")
        subprocess.run(['apt', '-y', 'install', '-qq', 'aria2'], check=True)

# GitHub에서 프로젝트 클론 함수
def clone_github_repo():
    """
    GitHub에서 프로젝트를 클론하고 폴더 이름을 변경합니다.
    """
    if not os.path.exists('./KiwKiw'):
        print("Cloning GitHub repository...")
        subprocess.run(['git', 'clone', 'https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git'], check=True)
        os.rename('./Retrieval-based-Voice-Conversion-WebUI', './KiwKiw')
    else:
        print("Repository already cloned.")

# 파일이 존재할 경우 삭제 함수
def remove_file_if_exists(file_path):
    """
    파일이 존재할 경우 삭제합니다.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed {file_path}")
    else:
        print(f"{file_path} does not exist.")

# aria2를 사용한 파일 다운로드 함수
def download_file(url, output_dir, output_file):
    """
    주어진 URL에서 파일을 다운로드하여 지정된 디렉토리에 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, output_file)
    aria2_command = ['aria2c', '--console-log-level=error', '-c', '-x', '16', '-s', '16', '-k', '1M', url, '-d', output_dir, '-o', output_file]
    
    if not os.path.exists(file_path):
        print(f"Downloading {output_file}...")
        subprocess.run(aria2_command, check=True)
    else:
        print(f"{output_file} already exists. Skipping download.")

# 미리 학습된 모델 다운로드 함수
def download_pretrained_models():
    """
    HuggingFace에서 미리 학습된 모델을 다운로드합니다.
    """
    pretrained_models = [
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D32k.pth', './KiwKiw/assets/pretrained', 'D32k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D40k.pth', './KiwKiw/assets/pretrained', 'D40k.pth'),
        # 필요한 추가 모델을 여기에 더할 수 있습니다.
    ]
    for url, output_dir, output_file in pretrained_models:
        download_file(url, output_dir, output_file)

# 최신 버전의 모델 다운로드 함수
def download_pretrained_v2_models():
    """
    HuggingFace에서 최신 버전의 미리 학습된 모델을 다운로드합니다.
    """
    pretrained_v2_models = [
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D32k.pth', './KiwKiw/assets/pretrained_v2', 'D32k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G32k.pth', './KiwKiw/assets/pretrained_v2', 'G32k.pth'),
        # 필요한 추가 모델을 여기에 더할 수 있습니다.
    ]
    for url, output_dir, output_file in pretrained_v2_models:
        download_file(url, output_dir, output_file)

# Hubert 모델 및 추가 자원 다운로드
def download_additional_assets():
    """
    추가적으로 필요한 Hubert 및 RMVPE 모델을 다운로드합니다.
    """
    additional_assets = [
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt', './KiwKiw/assets/hubert', 'hubert_base.pt'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt', './KiwKiw/assets/rmvpe', 'rmvpe.pt')
    ]
    for url, output_dir, output_file in additional_assets:
        download_file(url, output_dir, output_file)

# 압축 및 로컬에 파일 저장
def zip_and_save_locally():
    """
    작업 결과를 zip 파일로 압축하여 로컬 파일 시스템에 저장합니다.
    """
    zip_file = './KiwKiw.zip'
    remove_file_if_exists(zip_file)
    
    print("Zipping the project folder...")
    subprocess.run(['zip', '-r', 'KiwKiw.zip', 'KiwKiw'], check=True)
    print(f"Zip file saved locally at {zip_file}")

# 메인 실행 함수
def main():
    """
    전체 다운로드 파이프라인을 실행합니다.
    """
    install_aria2()
    clone_github_repo()
    download_pretrained_models()
    download_pretrained_v2_models()
    download_additional_assets()
    zip_and_save_locally()

# 실행
if __name__ == "__main__":
    main()
