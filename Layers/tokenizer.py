from transformer import AutoTokenizer
torch.nn.functional as F
def get_input_ids():
    tokenizer = AutoTokenizer.from_pretrained("/home/yash/gemma-4-E2B")
    input = "the quick brown fox jumps over the lazy dog"
    token_ids = tokenizer.encode(input)
    
def get_embeds(token_ids, embeddings):
    input_embeddings = F.embeddings(token_ids)



