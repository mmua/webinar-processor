import shutil
import subprocess
from typing import List
import os

def get_wav_filename(input_path: str, output_dir: str) -> str:
    """
    Generate a WAV filename from an input audio/video file path.
    
    Args:
        input_path: Path to the input audio/video file
        output_dir: Directory where the WAV file should be placed
        
    Returns:
        Path to the output WAV file
    """
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    # Create the WAV filename
    wav_filename = f"{base_name}.wav"
    return os.path.join(output_dir, wav_filename)

def convert_mp4_to_wav(input_path: str, output_path: str, sample_rate: int = 16000):
    cmd = [
        "ffmpeg", 
        "-i", input_path, 
        "-ar", str(sample_rate), 
        output_path
    ]
    subprocess.run(cmd, check=True)

def get_non_silence_intervals(input_path: str) -> List[float]:
    import re
    import subprocess
    import moviepy.editor as mpy
    
    # Use ffmpeg to detect silence
    command = ["ffmpeg", "-i",input_path, "-af", "silencedetect=noise=-30dB:d=5", "-nostats", "-f", "null", "-"]
    output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)

    # Parse the output to get silence start and end times
    pattern = r"silence_(start|end): (\d+(\.\d+)?)"
    matches = re.findall(pattern, output)
    times = [float(time) for _, time, _ in matches]
    
    if not times: # no long silence intervals
        return None

    # If the video starts with silence, start from the end of silence
    if times[0] == 0:
        times.pop(0)
    else: # start from beginning
        times.insert(0, 0) 

    # If the video doesn't end with silence, end at the last frame
    if len(times) % 2 != 0:
        clip = mpy.VideoFileClip(input_path)
        if times[-1] + 5 < clip.duration:
            times.append(clip.duration)
        else:
            times.pop(-1)
    
    return times

def mp4_silence_remove(input_path: str, output_path: str):
    import moviepy.editor as mpy

    times = get_non_silence_intervals(input_path)

    if times:
        clip = mpy.VideoFileClip(input_path)
        # Create a list of video clips that are not silence
        clips = [clip.subclip(start, end) for start, end in zip(times[::2], times[1::2]) if end - start > 1]

        # Concatenate the clips and write the result to a file
        final_clip = mpy.concatenate_videoclips(clips)
        final_clip.write_videofile(output_path)
    else:
        shutil.copyfile(input_path, output_path)

def wav_silence_remove(input_path: str, output_path: str, min_silence_len: int = 500, silence_thresh: int = -40):
    """
    Remove silence from a wav file and save the result.
    Args:
        input_path: Path to the input wav file.
        output_path: Path to save the output wav file.
        min_silence_len: Minimum length of silence to detect (in ms).
        silence_thresh: Silence threshold in dBFS.
    """
    from pydub import AudioSegment, silence
    import shutil

    audio = AudioSegment.from_wav(input_path)
    # Detect non-silent intervals (returns list of [start, end] in ms)
    nonsilent_ranges = silence.detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    if nonsilent_ranges:
        # Concatenate non-silent segments
        non_silent_audio = AudioSegment.empty()
        for start, end in nonsilent_ranges:
            non_silent_audio += audio[start:end]
        non_silent_audio.export(output_path, format="wav")
    else:
        shutil.copyfile(input_path, output_path)
