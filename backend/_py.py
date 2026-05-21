import cv2
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()
STREAM_KEY = os.getenv('cloudfare_key')
RTMPS_URL  = f"rtmps://live.cloudflare.com:443/live/{STREAM_KEY}"

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)   # changed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)    # changed
cap.set(cv2.CAP_PROP_FPS,          30)     # changed

WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS    = int(cap.get(cv2.CAP_PROP_FPS))

print(f"[CAMERA] {WIDTH}x{HEIGHT} @ {FPS}fps")

ffmpeg = subprocess.Popen([
    "ffmpeg", "-loglevel", "warning",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{WIDTH}x{HEIGHT}",
    "-r", str(FPS),
    "-i", "pipe:0",
    "-vcodec", "libx264",
    "-pix_fmt", "yuv420p",
    "-preset", "veryfast",
    "-tune", "zerolatency",
    "-profile:v", "main",        # main is fine for 720p
    "-level", "3.1",             # correct level for 720p30
    "-b:v", "4000k",             # 4Mbps — right for 720p30
    "-maxrate", "4000k",
    "-bufsize", "8000k",
    "-g", str(FPS),
    "-keyint_min", str(FPS),
    "-sc_threshold", "0",
    "-bf", "0",
    "-f", "flv",
    RTMPS_URL,
], stdin=subprocess.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    ffmpeg.stdin.write(frame.tobytes())

cap.release()
ffmpeg.stdin.close()
