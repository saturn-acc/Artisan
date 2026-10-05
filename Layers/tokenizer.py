from transformer import AutoTokenizer 

def get_input_ids():
    tokenizer = AutoTokenizer.from_pretrained("/home/yash/gemma-4-E2B")
    input = input(">")
    token_ids = tokenizer.encode(input)
    
def get_embeds(token_ids):
    :
