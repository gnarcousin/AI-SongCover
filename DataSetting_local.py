import os
import numpy as np
import librosa
import soundfile as sf
import subprocess
import webrtcvad

# 사용자 입력
mode = "Splitting"  # "Separate" 또는 "Splitting" 선택
audio_name = "IU_Live"  # 처리할 오디오 파일 이름
audio_input_path = os.path.join(".", f"{audio_name}.wav")  # 로컬에서 사용할 WAV 파일 경로

# 결과 저장 폴더 생성
output_folder = os.path.join(".", "processed_audio", audio_name)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Demucs로 보컬 분리 함수
def separate_audio(audio_path, output_folder):
    """Demucs를 사용하여 보컬과 악기 분리"""
    command = f"demucs --two-stems=vocals {audio_path} -o {output_folder}"
    subprocess.run(command.split(), stdout=subprocess.PIPE)
    print(f"Demucs 실행 완료. 파일이 저장된 경로: {output_folder}")

# VAD(Voice Activity Detection)를 사용한 음성 슬라이싱 클래스
class VAD_Slicer:
    def __init__(self, sr, mode=3, frame_duration=30):
        """초기화: WebRTC VAD 설정"""
        self.vad = webrtcvad.Vad(mode)  # 모드: 0 (낮은 민감도) ~ 3 (높은 민감도)
        self.sr = sr
        self.frame_duration = frame_duration  # 밀리초 단위 프레임 지속 시간
        self.frame_size = int(sr * frame_duration / 1000)  # 프레임 크기

    def is_speech(self, frame):
        """주어진 프레임이 음성인지 확인"""
        return self.vad.is_speech(frame.tobytes(), self.sr)

    def slice(self, waveform):
        """음성을 프레임 단위로 나누어 음성 구간만 반환"""
        chunks = []
        current_chunk = []
        for i in range(0, len(waveform), self.frame_size):
            frame = waveform[i:i+self.frame_size]
            if len(frame) < self.frame_size:
                break
            if self.is_speech(frame):
                current_chunk.append(frame)
            elif current_chunk:
                chunks.append(np.concatenate(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append(np.concatenate(current_chunk))
        return chunks

def process_audio():
    """오디오 처리 메인 함수"""
    # 보컬 분리
    print(f"Processing audio: {audio_input_path}")
    separate_audio(audio_input_path, output_folder)

    # 분리된 보컬 오디오 파일 경로
    vocals_path = os.path.join(output_folder, "htdemucs", audio_name, "vocals.wav")
    
    # Demucs가 파일을 제대로 생성했는지 확인
    if not os.path.exists(vocals_path):
        print(f"파일을 찾을 수 없습니다: {vocals_path}")
        return
    
    # 오디오 파일 로드
    try:
        audio, sr = librosa.load(vocals_path, sr=None, mono=False)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {vocals_path}")
        return

    # VAD를 사용하여 음성 구간 나누기
    if mode == "Splitting":
        slicer = VAD_Slicer(sr=sr)
        chunks = slicer.slice(audio)

        # 나누어진 음성 파일 저장
        for i, chunk in enumerate(chunks):
            if chunk.ndim > 1:
                chunk = chunk.T  # 스테레오 파일을 변환
            output_path = os.path.join(output_folder, f'split_{i}.wav')
            sf.write(output_path, chunk, sr)
            print(f"Saved chunk: {output_path}")
    else:
        print("Separate 모드에서는 추가 작업이 없습니다.")

if __name__ == "__main__":
    process_audio()