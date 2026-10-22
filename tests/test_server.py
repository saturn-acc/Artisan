import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from server import create_server


class InferenceAPITests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def generate(messages, specdecode=False):
            self.calls.append((messages, specdecode))
            return 'Backend reply'
        self.server = create_server(0, generate, api_key='test-key')
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()

    def request(self, body, token='test-key'):
        request = Request(self.url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
        return urlopen(request)

    def test_messages_and_decoding_reach_model(self):
        body = {'model': 'artisan', 'messages': [{'role': 'user', 'content': 'Hello'}], 'specdecode': True}
        with self.request(body) as response:
            self.assertEqual(json.load(response)['choices'][0]['message']['content'], 'Backend reply')
        self.assertEqual(self.calls, [(body['messages'], True)])
        body.pop('specdecode')
        with self.request(body):
            pass
        self.assertFalse(self.calls[-1][1])

    def test_invalid_request_and_auth_do_not_run_inference(self):
        good = {'messages': [{'role': 'user', 'content': 'Hello'}]}
        for bad in ({}, [], dict(good, stream=True), dict(good, specdecode='true')):
            with self.assertRaises(HTTPError) as result:
                self.request(bad)
            self.assertEqual(result.exception.code, 400)
            result.exception.close()
        with self.assertRaises(HTTPError) as result:
            self.request(good, token='wrong')
        self.assertEqual(result.exception.code, 401)
        result.exception.close()
        self.assertEqual(self.calls, [])
