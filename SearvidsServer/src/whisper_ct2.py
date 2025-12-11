# Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
# All rights reserved.

import sys
import os
import math
from faster_whisper import WhisperModel

def format_timestamp(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python whisper_ct2.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    # Default to base, user can set env
    model_size = os.environ.get("WHISPER_MODEL_SIZE", "base") 
    
    # Check for CUDA
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env is None or cuda_env == "":
        device = "cpu"
        compute_type = "int8" # CPU usually supports int8 better for speed
    else:
        device = "cuda"
        compute_type = "float16"

    try:
        # download_root can be used to store models in /app/models
        model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root="/app/models/faster_whisper")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    segments, info = model.transcribe(audio_file, beam_size=5)

    for segment in segments:
        start_str = format_timestamp(segment.start)
        end_str = format_timestamp(segment.end)
        # Output format matching the C++ regex: [00:00:00.000 --> 00:00:05.000] Text
        print(f"[{start_str} --> {end_str}] {segment.text}")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
