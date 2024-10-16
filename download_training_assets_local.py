# -*- coding: utf-8 -*-
"""
훈련을 위한 미리 학습된 모델 파일을 다운로드하고, 로컬 파일 시스템에 저장하는 스크립트.
"""
import os
import shutil
import stat
import subprocess

def remove_readonly(func, path, _):
    """
    읽기 전용 파일을 삭제할 때 사용되는 함수.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_github_repo():
    """
    GitHub에서 프로젝트를 클론하고 폴더 이름을 변경합니다.
    클론된 폴더를 복사하여 권한 문제를 방지합니다.
    """
    repo_url = 'https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git'
    repo_name = 'Retrieval-based-Voice-Conversion-WebUI'
    target_dir = 'KiwKiw'

    # GitHub에서 클론
    if not os.path.exists(repo_name):
        print("GitHub 레포지토리를 클론 중입니다...")
        subprocess.run(['git', 'clone', repo_url], check=True)
    else:
        print(f"{repo_name} 폴더가 이미 존재합니다. 클론을 건너뜁니다.")

    # 폴더 이름을 변경하기 위한 복사 과정
    if not os.path.exists(target_dir):
        print(f"{repo_name} 폴더를 {target_dir}로 복사 중입니다...")
        shutil.copytree(repo_name, target_dir)
        print(f"{repo_name} 폴더를 {target_dir}로 복사 완료.")
    else:
        print(f"{target_dir} 폴더가 이미 존재합니다. 복사를 건너뜁니다.")

    # 필요에 따라 기존 폴더 삭제
    if os.path.exists(repo_name):
        print(f"기존 {repo_name} 폴더를 삭제 중입니다...")
        shutil.rmtree(repo_name, onerror=remove_readonly)
        print(f"{repo_name} 폴더가 삭제되었습니다.")
# 파일 다운로드 함수
def download_file(url, output_dir, output_file):
    """
    주어진 URL에서 파일을 다운로드하여 지정된 디렉토리에 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, output_file)
    aria2_command = ['C:/ProgramData/aria2-1.37.0-win-64bit-build1/aria2c.exe', '--console-log-level=error', '-c', '-x', '16', '-s', '16', '-k', '1M', url, '-d', output_dir, '-o', output_file]
    
    if not os.path.exists(file_path):
        print(f"{output_file} 다운로드 중...")
        subprocess.run(aria2_command, check=True)
    else:
        print(f"{output_file} 파일이 이미 존재합니다. 다운로드를 건너뜁니다.")

# 미리 학습된 모델 다운로드
def download_pretrained_models():
    """
    HuggingFace에서 미리 학습된 모델을 다운로드합니다.
    """
    pretrained_models = [
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D32k.pth', './KiwKiw/assets/pretrained', 'D32k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D40k.pth', './KiwKiw/assets/pretrained', 'D40k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D48k.pth', './KiwKiw/assets/pretrained', 'D48k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G32k.pth', './KiwKiw/assets/pretrained', 'G32k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G40k.pth', './KiwKiw/assets/pretrained', 'G40k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G48k.pth', './KiwKiw/assets/pretrained', 'G48k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D32k.pth', './KiwKiw/assets/pretrained', 'f0D32k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D40k.pth', './KiwKiw/assets/pretrained', 'f0D40k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D48k.pth', './KiwKiw/assets/pretrained', 'f0D48k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G32k.pth', './KiwKiw/assets/pretrained', 'f0G32k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G40k.pth', './KiwKiw/assets/pretrained', 'f0G40k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G48k.pth', './KiwKiw/assets/pretrained', 'f0G48k.pth'),
    ]
    for url, output_dir, output_file in pretrained_models:
        download_file(url, output_dir, output_file)

# RVC V2 모델 다운로드
def download_pretrained_v2_models():
    """
    HuggingFace에서 최신 버전의 미리 학습된 모델을 다운로드합니다.
    """
    pretrained_v2_models = [
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D32k.pth', './KiwKiw/assets/pretrained_v2', 'D32k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth', './KiwKiw/assets/pretrained_v2', 'D40k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D48k.pth', './KiwKiw/assets/pretrained_v2', 'D48k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G32k.pth', './KiwKiw/assets/pretrained_v2', 'G32k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth', './KiwKiw/assets/pretrained_v2', 'G40k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G48k.pth', './KiwKiw/assets/pretrained_v2', 'G48k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D32k.pth', './KiwKiw/assets/pretrained_v2', 'f0D32k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth', './KiwKiw/assets/pretrained_v2', 'f0D40k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth', './KiwKiw/assets/pretrained_v2', 'f0D48k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G32k.pth', './KiwKiw/assets/pretrained_v2', 'f0G32k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth', './KiwKiw/assets/pretrained_v2', 'f0G40k.pth'),
        ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth', './KiwKiw/assets/pretrained_v2', 'f0G48k.pth'),
    ]
    for url, output_dir, output_file in pretrained_v2_models:
        download_file(url, output_dir, output_file)

# 추가 자원 다운로드 (hubert, rmvpe)
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
    if os.path.exists(zip_file):
        os.remove(zip_file)
    
    print("프로젝트 폴더를 압축하는 중...")
    shutil.make_archive('KiwKiw', 'zip', './KiwKiw')
    print(f"Zip 파일이 로컬에 저장되었습니다: {zip_file}")

# 메인 실행 함수
def main():
    """
    전체 다운로드 파이프라인을 실행합니다.
    """
    clone_github_repo()
    download_pretrained_models()
    download_pretrained_v2_models()
    download_additional_assets()
    zip_and_save_locally()

# 실행
if __name__ == "__main__":
    main()
