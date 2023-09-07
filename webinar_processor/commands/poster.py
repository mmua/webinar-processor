"""
Extract poster from video
"""
import click
import cv2
import numpy as np
import subprocess

def extract_frames(video_path: str, poster_path: str, interval=10):
    # Extract frames at regular intervals using ffmpeg
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'select=not(mod(n\,{interval}))',
        '-vsync', 'vfr',
        '-q:v', '2',
        f'{poster_path}/tmp_%03d.jpg'
    ]
    subprocess.call(cmd)
    return [f'{poster_path}/tmp_{i:03d}.jpg' for i in range(1, 1000) if cv2.imread(f'{poster_path}/tmp_{i:03d}.jpg', cv2.IMREAD_UNCHANGED) is not None] # Adjust range accordingly

def detect_scene_change(frame1, frame2, threshold=0.3):
    # Calculate the absolute difference between two frames
    diff = cv2.absdiff(frame1, frame2)
    mean_diff = np.mean(diff)
    return mean_diff > threshold

def detect_face(frame, face_cascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def select_poster(video_path, poster_path):
    frames = extract_frames(video_path, poster_path)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    prev_frame = None
    for frame_path in frames[len(frames) // 4: ]:
        current_frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                
        # scene change without faces, prioritize
        if prev_frame is not None and detect_scene_change(prev_frame, current_frame) and not detect_face(current_frame, face_cascade):
            return frame_path
        
        prev_frame = current_frame

    # If no suitable frame found, return the one in the first quater
    return frames[len(frames) // 4]

@click.command()
@click.argument('video_path', nargs=1)
@click.argument('poster_path', nargs=1)
def poster(video_path: str, poster_path: str):
    """
    Extract poster from video
    """
    poster_path = select_poster(video_path, poster_path)
    print(f"Selected poster: {poster_path}")