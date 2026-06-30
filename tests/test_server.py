"""Tests for the HTTP inference server.

Spins up the server in a background thread using random weights (no
checkpoint needed), sends real HTTP requests, and verifies responses.
The server is started once for the module and torn down after all tests.

Three things verified:
  1. /health returns {"status": "ok"}.
  2. /generate returns the correct shape and content (prompt preserved,
     correct number of generated tokens).
  3. /generate output matches direct generate() -- the HTTP layer must
     not change the computation.
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server.server as srv
from model.config import LlamaConfig
from model.model import generate
from testutil.random_weights import random_model_weights

# ---------------------------------------------------------------------------
# Module-level server fixture
# ---------------------------------------------------------------------------

_PORT = 18765  # Use a non-standard port to avoid conflicts


def _wait_for_server(port: int, retries: int = 20, delay: float = 0.1):
    for _ in range(retries):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError(f"Server on port {port} did not start in time")


def setup_module(module):
    """Start the inference server once for all tests in this module."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=50, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        intermediate_size=32, rms_norm_eps=1e-5, rope_theta=10000.0,
    )
    weights = random_model_weights(config, num_layers=config.num_hidden_layers)

    srv._config = config
    srv._scheduler = srv.make_scheduler(
        config, weights, total_blocks=128, block_size=4, dtype=torch.float32
    )

    httpd = HTTPServer(("127.0.0.1", _PORT), srv.InferenceHandler)
    module._httpd = httpd

    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    _wait_for_server(_PORT)


def teardown_module(module):
    module._httpd.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(path: str, payload: dict) -> tuple:
    """Returns (status_code, response_dict)."""
    url = f"http://127.0.0.1:{_PORT}{path}"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(path: str) -> tuple:
    url = f"http://127.0.0.1:{_PORT}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    status, body = _get("/health")
    assert status == 200
    assert body == {"status": "ok"}


def test_unknown_get_returns_404():
    status, body = _get("/notapath")
    assert status == 404


def test_unknown_post_returns_404():
    status, body = _post("/notapath", {})
    assert status == 404


def test_generate_returns_200_with_token_ids():
    status, body = _post("/generate", {"token_ids": [1, 2, 3], "max_new_tokens": 4})
    assert status == 200
    assert "token_ids" in body
    assert isinstance(body["token_ids"], list)


def test_generate_preserves_prompt():
    prompt = [5, 6, 7, 8]
    status, body = _post("/generate", {"token_ids": prompt, "max_new_tokens": 3})
    assert status == 200
    assert body["token_ids"][:4] == prompt


def test_generate_returns_correct_total_length():
    prompt = [1, 2, 3]
    max_new = 5
    status, body = _post("/generate", {"token_ids": prompt, "max_new_tokens": max_new})
    assert status == 200
    assert len(body["token_ids"]) == len(prompt) + max_new


def test_generate_missing_token_ids_returns_400():
    status, body = _post("/generate", {"max_new_tokens": 5})
    assert status == 400


def test_generate_empty_token_ids_returns_400():
    status, body = _post("/generate", {"token_ids": [], "max_new_tokens": 5})
    assert status == 400


def test_generate_missing_max_new_tokens_returns_400():
    status, body = _post("/generate", {"token_ids": [1, 2, 3]})
    assert status == 400


def test_generate_invalid_json_returns_400():
    url = f"http://127.0.0.1:{_PORT}/generate"
    req = urllib.request.Request(
        url, data=b"not json", headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    assert status == 400


def test_generate_output_matches_direct_generate():
    # The HTTP server must produce the same token sequence as calling
    # generate() directly with the same weights and prompt.
    # We re-create the same weights using the same seed as setup_module.
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=50, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        intermediate_size=32, rms_norm_eps=1e-5, rope_theta=10000.0,
    )
    weights = random_model_weights(config, num_layers=config.num_hidden_layers)

    prompt = [1, 2, 3, 4]
    max_new = 5

    direct = generate(
        torch.tensor([prompt], dtype=torch.long), weights, config, max_new_tokens=max_new
    )[0].tolist()

    status, body = _post("/generate", {"token_ids": prompt, "max_new_tokens": max_new})
    assert status == 200
    assert body["token_ids"] == direct, (
        f"Server output differs from direct generate().\n"
        f"Direct: {direct}\nServer: {body['token_ids']}"
    )
