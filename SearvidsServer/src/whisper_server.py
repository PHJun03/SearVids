import sys
import os
import math
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

app = Flask(__name__)

# [Configuration]
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
DEVICE = "cuda" if CUDA_VISIBLE_DEVICES else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
MODEL_PATH = "/app/models/faster_whisper"

print(f"[WhisperServer] Initializing... (Device: {DEVICE}, Model: {MODEL_SIZE})", flush=True)

# Load Model Once
try:
    if os.path.exists(f"{MODEL_PATH}/{MODEL_SIZE}"):
        print(f"[WhisperServer] Loading local model from {MODEL_PATH}/{MODEL_SIZE}...", flush=True)
        model = WhisperModel(f"{MODEL_PATH}/{MODEL_SIZE}", device=DEVICE, compute_type=COMPUTE_TYPE)
    else:
        print(f"[WhisperServer] Downloading and loading model {MODEL_SIZE}...", flush=True)
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_PATH)
    print("[WhisperServer] Model loaded successfully!", flush=True)
except Exception as e:
    print(f"[WhisperServer] Failed to load model: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

def format_timestamp(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.json
        if not data or 'file_path' not in data:
            return jsonify({"error": "No file_path provided"}), 400
        
        file_path = data['file_path']
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404

        print(f"[WhisperServer] Transcribing: {file_path}", flush=True)
        
        # Run inference
        segments, info = model.transcribe(file_path, beam_size=5, vad_filter=True)
        
        # Stream results as a single string (to match old output format) or JSON list
        # To match C++ regex parsing: [00:00:00.000 --> 00:00:05.000] Text
        output_lines = []
        for segment in segments:
            start_str = format_timestamp(segment.start)
            end_str = format_timestamp(segment.end)
            line = f"[{start_str} --> {end_str}] {segment.text}"
            output_lines.append(line)
        
        full_text = "\n".join(output_lines)
        return jsonify({"transcript": full_text})

    except Exception as e:
        print(f"[WhisperServer] Error: {e}", file=sys.stderr, flush=True)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": MODEL_SIZE})

if __name__ == '__main__':
    # Run on port 5000 inside the container
    app.run(host='0.0.0.0', port=5000)
