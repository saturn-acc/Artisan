"""Local Artisan chat: python3 chat.py, then open the printed URL."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from model import Model

specdecode = False


class Chat:
    def __init__(self, generate=None, port=8765):
        self.generate = generate if generate is not None else Model()
        self.messages = []
        self.port = port
        self.lock = Lock()

    def submit(self, text):
        global specdecode
        text = text.strip()
        if not text:
            raise ValueError('Enter a message first.')
        command = ' '.join(text.lower().split())
        if command in ('/speculative decoding', '/speculative decoding off'):
            specdecode = command == '/speculative decoding'
            if hasattr(self.generate, 'specdecode'):
                self.generate.specdecode = specdecode
            return {'kind': 'notice', 'text': 'Speculative decoding ' + ('enabled.' if specdecode else 'disabled.'), 'specdecode': specdecode}
        if text.startswith('/'):
            return {'kind': 'notice', 'text': 'Use /speculative decoding or /speculative decoding off.', 'specdecode': specdecode}
        pending = self.messages + [{'role': 'user', 'content': text}]
        reply = str(self.generate(pending, specdecode=specdecode))
        self.messages = pending + [{'role': 'assistant', 'content': reply}]
        return {'kind': 'assistant', 'text': reply, 'specdecode': specdecode}

    def run(self):
        chat = self
        page = Path(__file__).parent / 'ui' / 'index.html'

        class Handler(BaseHTTPRequestHandler):
            def respond(self, status, payload, content_type='application/json'):
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == '/':
                    self.respond(200, page.read_bytes(), 'text/html; charset=utf-8')
                elif self.path == '/state':
                    with chat.lock:
                        self.respond(200, {'messages': chat.messages, 'specdecode': specdecode})
                else:
                    self.respond(404, {'error': 'Not found'})

            def do_POST(self):
                if self.path not in ('/chat', '/new-chat'):
                    self.respond(404, {'error': 'Not found'})
                    return
                # Only this local page may submit requests from a browser.
                origin = self.headers.get('Origin')
                allowed = {f'http://127.0.0.1:{chat.port}', f'http://localhost:{chat.port}'}
                if origin not in allowed or self.headers.get_content_type() != 'application/json':
                    self.respond(403, {'error': 'Open the chat using its local URL.'})
                    return
                try:
                    length = int(self.headers.get('Content-Length', 0))
                    if not 0 < length <= 65536:
                        raise ValueError('Message is too large or empty.')
                    body = json.loads(self.rfile.read(length))
                    if self.path == '/new-chat':
                        with chat.lock:
                            chat.messages.clear()
                            if hasattr(chat.generate, 'last_input'):
                                chat.generate.last_input = None
                        self.respond(200, {'specdecode': specdecode})
                        return
                    text = body.get('text')
                    if not isinstance(text, str):
                        raise ValueError('Message must be text.')
                    with chat.lock:
                        result = chat.submit(text)
                    self.respond(200, result)
                except Exception as error:
                    self.respond(400, {'error': str(error)})

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', self.port), Handler)
        print(f'Artisan chat → http://127.0.0.1:{self.port}', flush=True)
        print('Press Ctrl+C to stop.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    Chat().run()
