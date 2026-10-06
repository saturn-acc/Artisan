from transformers import AutoTokenizer
import torch.nn.functional as F

def get_input_ids():
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/yash/gemma-4-E2B", local_files_only=True
    )
    input = "the quick brown fox jumps over the lazy dog"
    token_ids = tokenizer.encode(input)
    return token_ids
    
def get_embeds(token_ids, embeddings):
    input_embeddings = F.embedding(token_ids, embeddings)



