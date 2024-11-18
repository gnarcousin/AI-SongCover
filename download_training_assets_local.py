import os
import subprocess
from pathlib import Path

# 1. GitHub 저장소 클론
def clone_repository():
    if not os.path.exists("KiwKiw"):
        print("Cloning KiwKiw repository...")
        subprocess.run(["git", "clone", "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git", "KiwKiw"])
    else:
        print("KiwKiw repository already exists.")

# 2. 모델 파일 다운로드
def download_model_files():
    model_urls = [
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D32k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D40k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D48k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G32k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G40k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G48k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D32k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D40k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D48k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G32k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G40k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G48k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D32k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D48k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G32k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G48k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D32k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G32k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    ]

    base_dirs = {
        "pretrained": "KiwKiw/assets/pretrained",
        "pretrained_v2": "KiwKiw/assets/pretrained_v2",
        "hubert": "KiwKiw/assets/hubert",
        "rmvpe": "KiwKiw/assets/rmvpe",
    }

    for url in model_urls:
        filename = url.split("/")[-1]
        category = "pretrained"
        if "pretrained_v2" in url:
            category = "pretrained_v2"
        elif "hubert" in url:
            category = "hubert"
        elif "rmvpe" in url:
            category = "rmvpe"
        save_path = os.path.join(base_dirs[category], filename)
        os.makedirs(base_dirs[category], exist_ok=True)
        if not os.path.exists(save_path):
            print(f"Downloading {filename}...")
            subprocess.run(["aria2c", "-c", "-x", "16", "-s", "16", "-k", "1M", url, "-d", base_dirs[category], "-o", filename])
        else:
            print(f"{filename} already exists. Skipping download.")

import shutil

def zip_files():
    output_zip = "KiwKiw.zip"
    source_dir = "KiwKiw"
    
    if os.path.exists(output_zip):
        os.remove(output_zip)

    print(f"Zipping {source_dir} into {output_zip}...")
    shutil.make_archive("KiwKiw", 'zip', source_dir)
    
# 메인 실행
if __name__ == "__main__":
    # clone_repository()
    # download_model_files()
    zip_files()
    print("All tasks completed successfully.")