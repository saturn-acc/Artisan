"""Local inference API. Run python3 server.py; connect a separate chat frontend."""
import argparse
import hmac
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from model import Model

MAX_BODY = 2 * 1024 * 1024


def create_server(port=8000, generate=None, api_key=None):
    model = generate if generate is not None else Model()
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            if api_key and not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + api_key):
                self.respond(401, {'error': {'message': 'Invalid API key.'}})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == '/health':
                self.respond(200, {'status': 'ok'})
            elif self.path == '/v1/models':
                self.respond(200, {'object': 'list', 'data': [{'id': 'artisan', 'object': 'model', 'owned_by': 'local'}]})
            else:
                self.respond(404, {'error': {'message': 'Not found.'}})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != '/v1/chat/completions':
                self.respond(404, {'error': {'message': 'Not found.'}})
                return
            # The separate frontend's Python client calls this API. Browser
            # origins have no reason to send requests directly to inference.
            if self.headers.get('Origin') or self.headers.get_content_type() != 'application/json':
                self.respond(403, {'error': {'message': 'Use a server-side JSON client.'}})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= MAX_BODY:
                    raise ValueError('Request is empty or too large.')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON object.')
                messages = body.get('messages')
                if not isinstance(messages, list) or not messages:
                    raise ValueError('messages must be a non-empty array.')
                for message in messages:
                    if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant') or not isinstance(message.get('content'), str):
                        raise ValueError('Each message needs a role and text content.')
                if messages[-1]['role'] != 'user' or not messages[-1]['content'].strip():
                    raise ValueError('The last message must contain user text.')
                if body.get('model', 'artisan') != 'artisan':
                    raise ValueError('Unknown model. Use artisan.')
                if body.get('stream', False) is not False:
                    raise ValueError('Streaming is not implemented; use stream=false.')
                specdecode = body.get('specdecode', False)
                if not isinstance(specdecode, bool):
                    raise ValueError('specdecode must be a boolean.')
            except (ValueError, UnicodeError) as error:
                self.respond(400, {'error': {'message': str(error)}})
                return
            try:
                with lock:
                    text = str(model(messages, specdecode=specdecode))
            except Exception as error:
                self.respond(500, {'error': {'message': str(error)}})
                return
            self.respond(200, {
                'id': 'chatcmpl-' + uuid.uuid4().hex,
                'object': 'chat.completion', 'created': int(time.time()), 'model': 'artisan',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
            })

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Artisan local inference API')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    try:
        server = create_server(args.port, api_key=os.environ.get('ARTISAN_API_KEY'))
    except OSError as error:
        parser.error(str(error))
    print(f'Artisan inference → http://127.0.0.1:{server.server_port}/v1', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
