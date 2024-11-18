import os
import librosa
import soundfile as sf
import numpy as np
import subprocess
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 작업 디렉토리 설정
BASE_DIR = "C:/Users/한윤택.HYT/Desktop/last"
DATASET_DIR = f"{BASE_DIR}/madesong"
YOUTUBE_DIR = f"{BASE_DIR}/youtubeaudio"
SEPARATED_DIR = f"{BASE_DIR}/separated"
SPLITTED_DIR = f"{BASE_DIR}/splitted"

# 필요한 폴더 생성
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(YOUTUBE_DIR, exist_ok=True)
os.makedirs(SEPARATED_DIR, exist_ok=True)
os.makedirs(SPLITTED_DIR, exist_ok=True)

# 음성 파일 다운로드 또는 로드 설정
MODE = "Local"  # "Youtube" 또는 "Local"
AUDIO_NAME = "TestCase"  # 음성 파일 이름 (확장자 제외)
YOUTUBE_URL = "https://www.youtube.com/watch?v=Ml2w_c87UJw"  # 유튜브 URL (Youtube 모드인 경우)
LOCAL_AUDIO_PATH = f"{DATASET_DIR}/{AUDIO_NAME}.wav"  # 저장된 음성 파일 경로 (Local 모드인 경우)


def download_youtube_audio(audio_name, youtube_url):
    """유튜브 음성 파일 다운로드"""
    print(f"Downloading audio from YouTube: {youtube_url}")
    try:
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bestaudio/best",
                "--extract-audio",
                "--audio-format",
                "wav",
                "--audio-quality",
                "0",
                "-o",
                f"{YOUTUBE_DIR}/{audio_name}.%(ext)s",
                youtube_url,
            ],
            check=True,
        )
        print(f"Downloaded audio saved to: {YOUTUBE_DIR}/{audio_name}.wav")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading audio: {e}")


def separate_audio(audio_input_path, output_dir):
    """Demucs를 사용하여 음성 파일 분리"""
    print(f"Separating vocals from audio: {audio_input_path}")
    try:
        subprocess.run(
            ["demucs", "--two-stems", "vocals", "-o", output_dir, audio_input_path],
            check=True,
        )
        print(f"Separated audio saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error separating audio: {e}")


def slice_audio(input_audio_path, output_dir):
    """음성 파일을 침묵 구간을0 기준으로 나눔"""
    print(f"Slicing audio into chunks: {input_audio_path}")
    audio, sr = librosa.load(input_audio_path, sr=None, mono=False)
    slicer = Slicer(
        sr=sr,
        threshold=-40,
        min_length=5000,
        min_interval=500,
        hop_size=10,
        max_sil_kept=500,
    )
    chunks = slicer.slice(audio)
    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T  # Transpose multi-channel audio
        output_file = f"{output_dir}/split_{i}.wav"
        sf.write(output_file, chunk, sr)
        print(f"Saved chunk to: {output_file}")


class Slicer:
    """음성 파일을 침묵 구간 기준으로 나누는 클래스"""
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError("The following condition must be satisfied: max_sil_kept >= hop_size")
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform

        rms_list = librosa.feature.rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0

        total_frames = rms_list.shape[0]
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue

            if silence_start == 0 and i > self.max_sil_kept:
                pos = rms_list[silence_start:i].argmin() + silence_start
                sil_tags.append((0, pos))
                clip_start = pos
            elif i - silence_start > self.max_sil_kept:
                pos = rms_list[silence_start:i].argmin() + silence_start
                sil_tags.append((clip_start, pos))
                clip_start = pos
            silence_start = None

        if silence_start is not None and total_frames - silence_start > self.min_length:
            pos = rms_list[silence_start:].argmin() + silence_start
            sil_tags.append((clip_start, pos))

        chunks = []
        for start, end in sil_tags:
            start_sample = start * self.hop_size
            end_sample = min(end * self.hop_size, len(samples))
            chunks.append(samples[start_sample:end_sample])

        return chunks


def main():
    if MODE == "Youtube":
        download_youtube_audio(AUDIO_NAME, YOUTUBE_URL)
        input_audio_path = f"{YOUTUBE_DIR}/{AUDIO_NAME}.wav"
    elif MODE == "Local":
        input_audio_path = LOCAL_AUDIO_PATH

    # Step 2: Separate vocals
    separate_audio(input_audio_path, SEPARATED_DIR)

    # Step 3: Slice the vocals into chunks
    vocals_path = f"{SEPARATED_DIR}/htdemucs/{AUDIO_NAME}/vocals.wav"
    slice_audio(vocals_path, f"{SPLITTED_DIR}/{AUDIO_NAME}")


if __name__ == "__main__":
    main()
