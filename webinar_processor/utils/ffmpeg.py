import subprocess

def convert_mp4_to_wav(input_path: str, output_path: str, sample_rate: int = 16000):
    cmd = [
        "ffmpeg", 
        "-i", input_path, 
        "-ar", str(sample_rate), 
        output_path
    ]
    subprocess.run(cmd, check=True)