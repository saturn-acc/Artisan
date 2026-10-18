"""Chat-to-model connection. Supply embedding weights and forward when ready."""
from Layers.tokenizer import get_input_ids, get_embeds


class Model:
    def __init__(self, embeddings=None, forward=None):
        self.embeddings = embeddings
        self.forward = forward
        self.specdecode = False
        self.last_input = None

    def __call__(self, messages, specdecode=False):
        self.specdecode = specdecode
        text = messages[-1]['content']
        self.last_input = {'text': text, 'token_ids': None,
                           'input_embeddings': None, 'specdecode': specdecode}
        token_ids = get_input_ids(text)
        self.last_input['token_ids'] = token_ids
        if self.embeddings is None:
            return f'Tokenized {len(token_ids)} tokens. Waiting for embedding weights.'
        input_embeddings = get_embeds(token_ids, self.embeddings)
        self.last_input['input_embeddings'] = input_embeddings
        if self.forward is None:
            return 'Embeddings ready. Waiting for the model forward function.'
        return self.forward(input_embeddings, specdecode=specdecode)
