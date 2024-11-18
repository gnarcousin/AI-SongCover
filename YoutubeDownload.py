import yt_dlp
import os
import ffmpeg

url= "https://www.youtube.com/watch?v=Ml2w_c87UJw"
Audio_Name = "TestCase"

os.mkdir("youtubeaudio")

ydl_opts = {
    'format' : 'bestaudio/best',
    'postprocessors': [{
        'key' : 'FFmpegExtractAudio',
        'preferredcodec' : 'wav',
    }],
    'outtmpl' : f'youtubeaudio/{Audio_Name}',
}

def download_from_url(url):
    ydl.download([url])

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    download_from_url(url)