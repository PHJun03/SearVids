# Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
# All rights reserved.

import sys
import os
# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import CLIPTokenizer
except ImportError:
    print("Error: transformers module not found. Please install it with 'pip install transformers'", file=sys.stderr)
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        # If no text provided, just exit (used for pre-loading/checking)
        return

    text = sys.argv[1]
    
    # Use openai/clip-vit-base-patch32 as it matches the standard CLIP model
    model_name = "openai/clip-vit-base-patch32"
    
    try:
        # This will download the tokenizer files to the cache directory on first run
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        # Tokenize
        # padding="max_length" ensures we get exactly max_length tokens
        # truncation=True ensures we don't exceed max_length
        # max_length=77 is standard for CLIP
        tokens = tokenizer(text, padding="max_length", max_length=77, truncation=True)
        
        # Output space-separated token IDs
        ids = tokens['input_ids']
        print(" ".join(map(str, ids)))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
