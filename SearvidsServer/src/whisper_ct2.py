# Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
# All rights reserved.

import sys
import os
import math
from faster_whisper import WhisperModel

# Force stdout to use UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

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
        # Check if local model exists
        local_model_path = f"/app/models/faster_whisper/{model_size}"
        if os.path.exists(local_model_path):
            print(f"Loading local Whisper model from {local_model_path}...")
            model = WhisperModel(local_model_path, device=device, compute_type=compute_type)
        else:
            print(f"Downloading Whisper model {model_size}...")
            # download_root can be used to store models in /app/models
            model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root="/app/models/faster_whisper")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Enable VAD filter (Silero VAD)
    # Adaptive Segment Filter (VAD + Terminal Backup)
    # We disable built-in VAD filter to see ALL segments, then manually filter.
    # Logic: Keep if (Speech Detected) OR (Time since last kept > 10.0s)
    segments, info = model.transcribe(audio_file, beam_size=5, vad_filter=False)

    last_kept_end = 0.0
    MIN_TERMINAL_DURATION = 10.0
    SPEECH_THRESHOLD = 0.5 # Threshold for no_speech_prob (lower means more likely speech)

    for segment in segments:
        # Check for speech
        # no_speech_prob is high if silence/noise.
        is_speech = segment.no_speech_prob < SPEECH_THRESHOLD
        
        # Check for adaptive terminal
        time_since_last = segment.start - last_kept_end
        is_terminal_reached = time_since_last >= MIN_TERMINAL_DURATION

        if is_speech or is_terminal_reached:
            # Keep this segment
            start_str = format_timestamp(segment.start)
            end_str = format_timestamp(segment.end)
            # Output format matching the C++ regex: [00:00:00.000 --> 00:00:05.000] Text
            print(f"[{start_str} --> {end_str}] {segment.text}")
            sys.stdout.flush()
            last_kept_end = segment.end
        else:
            # Drop segment (Silence/Noise and not enough time passed yet)
            pass

if __name__ == "__main__":
    main()
