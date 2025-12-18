# Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
# All rights reserved.

import sys
import os
# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers module not found. Please install it with 'pip install transformers'", file=sys.stderr)
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        # If no text provided, just exit (used for pre-loading/checking)
        return

    text = sys.argv[1]
    
    # Use local model path if available, otherwise fallback to Hub
    model_path = "/app/models/siglip_text"
    if not os.path.exists(model_path):
        model_path = "google/siglip-base-patch16-224"
    
    try:
        # This will download the tokenizer files to the cache directory on first run
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Tokenize
        # Do not pad to max_length. Let the C++ side handle the dynamic length.
        # This avoids polluting the embedding with padding tokens if the model 
        # doesn't accept an attention_mask (which seems to be the case here).
        tokens = tokenizer(text, truncation=True, max_length=64)
        
        # Output space-separated token IDs
        ids = tokens['input_ids']
        print(" ".join(map(str, ids)))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
