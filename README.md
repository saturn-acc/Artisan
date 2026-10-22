# Artisan

Local inference experiments: tokenization, embeddings, model execution hooks,
and Metal kernels. The chat and macOS notch UI now live in the separate private
[artisan-ui repository](https://github.com/saturn-acc/artisan-ui).

## Run the inference API

```sh
python3 -m pip install -r requirements.txt
export ARTISAN_MODEL_PATH=/path/to/your/model
python3 server.py
```

The API binds to `127.0.0.1:8000`. Change it with `--port`.
Optionally set `ARTISAN_API_KEY` to require a bearer token; set the same value in
`LLM_API_KEY` when launching Artisan UI.

Run the UI from its own checkout in a second terminal:

```sh
cd ../Artisan-UI
python3 chat.py -d
```

Omit `-d` for browser mode. The frontend sends full conversation history to the
API; this process owns model execution and has no browser or native UI dependency.

## API

- `GET /health`: process health (does not assert that weights are loaded).
- `GET /v1/models`: advertises the local `artisan` model.
- `POST /v1/chat/completions`: accepts `model`, `messages`, `stream: false`, and
  optional `specdecode: true` or `false` (default false).

Responses contain text at `choices[0].message.content`. Streaming is not yet
implemented. The API is intended for a local server-side client such as Artisan UI.

## Model connection

`model.py` owns the inference pipeline. Supply embedding weights to `Model` and
provide its `forward` callback to connect generation. Tokenization uses the local
model directory through `Layers/tokenizer.py`. Until weights and forward execution
are supplied, responses report the tokenization/embedding stage reached.
`specdecode` is passed to the pipeline; the flag alone does not implement decoding.

```sh
python3 -m unittest discover -s tests -v
```

The API tests use a fake model and do not require tokenizer files or weights.
