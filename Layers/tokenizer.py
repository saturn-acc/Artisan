"""Text preparation for the inference pipeline."""
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_tokenizer():
    path = Path(os.environ.get('ARTISAN_MODEL_PATH', '/home/yash/gemma-4-E2B')).expanduser()
    if not path.is_dir():
        raise FileNotFoundError('Tokenizer not configured. Set ARTISAN_MODEL_PATH to your local model directory.')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
    )


def get_input_ids(text):
    return load_tokenizer().encode(text)


def get_embeds(token_ids, embeddings):
    import torch
    import torch.nn.functional as F
    indices = torch.as_tensor(token_ids, dtype=torch.long, device=embeddings.device)
    return F.embedding(indices, embeddings)
