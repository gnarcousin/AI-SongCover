# 필수 라이브러리 설치 및 임포트
import os
import shutil
import numpy as np
import librosa
import soundfile as sf
from spleeter.separator import Separator
import webrtcvad
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 사용자로부터 음성 파일 경로를 입력받아 처리합니다.
# 예: 음성 파일이 /content/input_audio/sample.wav 위치에 있을 경우

audio_path = '/content/input_audio/sample.wav'  # 사용자가 제공한 음성 파일 경로
audio_name = 'sample'  # 음성 파일의 이름

# 출력 파일 저장 경로 설정
output_dir = f'/content/processed_audio/{audio_name}'
os.makedirs(output_dir, exist_ok=True)

# 1. 오디오 파일 로드
def load_audio(file_path):
    """
    주어진 경로에서 오디오 파일을 불러옵니다.
    """
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        logging.info(f"Loaded audio file: {file_path}")
        return audio, sr
    except Exception as e:
        logging.error(f"Error loading audio file: {e}")
        raise

# 2. Spleeter를 사용한 보컬 및 악기 분리
def separate_audio(input_path, output_path):
    """
    Spleeter를 사용하여 보컬과 악기를 분리하고, 분리된 파일을 저장합니다.
    """
    try:
        separator = Separator('spleeter:2stems')
        separator.separate_to_file(input_path, output_path)
        logging.info("Audio separation complete.")
    except Exception as e:
        logging.error(f"Error during audio separation: {e}")
        raise

# 3. 음성 슬라이싱 함수 정의 (VAD 사용)
def slice_audio_with_vad(audio, sr):
    """
    VAD(Voice Activity Detection)를 사용하여 음성 신호에서 침묵 구간을 감지하고, 침묵을 기준으로 음성을 슬라이싱합니다.
    """
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # 가장 공격적인 침묵 감지 모드
    frame_length = int(sr * 0.03)  # 30ms 프레임 길이
    hop_length = int(sr * 0.01)  # 10ms 간격으로 프레임 이동

    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    slices = []
    for i, frame in enumerate(frames.T):
        if vad.is_speech(frame.tobytes(), sr):
            start = i * hop_length
            end = start + frame_length
            slices.append(audio[start:end])
    logging.info("Audio slicing complete.")
    return slices

# 4. 슬라이싱된 오디오 파일 저장
def save_chunks(chunks, sr, output_dir, audio_name):
    """
    슬라이싱된 오디오 청크들을 개별 파일로 저장합니다.
    """
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T  # 오디오 타입 전환
        file_path = os.path.join(output_dir, f'split_{i}.wav')
        sf.write(file_path, chunk, sr)
        logging.info(f"Saved sliced audio chunk: {file_path}")

# 5. 메인 파이프라인: 전체 처리 흐름
def process_audio(audio_path, audio_name):
    """
    음성 파일을 로드하고, 보컬 분리, 슬라이싱 및 저장까지의 전체 프로세스를 수행합니다.
    """
    try:
        # 오디오 파일 로드
        audio, sr = load_audio(audio_path)
        
        # 오디오 분리 수행 (보컬/악기)
        separate_audio(audio_path, output_dir)
        
        # 분리된 보컬 오디오 파일 경로 설정
        vocal_path = f'{output_dir}/{audio_name}/vocals.wav'
        audio_vocal, sr_vocal = load_audio(vocal_path)
        
        # 슬라이싱 수행
        chunks = slice_audio_with_vad(audio_vocal, sr_vocal)
        
        # 슬라이싱된 파일 저장
        save_chunks(chunks, sr_vocal, output_dir, audio_name)
        logging.info(f"Audio processing complete. Processed files saved in {output_dir}")
    except Exception as e:
        logging.error(f"Error in audio processing: {e}")
        raise

# 프로세스 실행
process_audio(audio_path, audio_name)
