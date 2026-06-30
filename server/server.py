"""A minimal HTTP inference server built on Python's stdlib http.server.

No Flask, FastAPI, or external web framework -- just http.server from the
standard library. This is a deliberate scope choice: the interesting
engineering here is the inference stack (scheduler, paged cache, model),
not the web framework. Using stdlib keeps the dependency list short and
makes the server's mechanics fully visible.

Endpoints:
  POST /generate   -- submit a generation request, block until done,
                      return the generated tokens as JSON.
  GET  /health     -- returns {"status": "ok"} immediately. Used by tests
                      to wait for the server to be ready before sending
                      generation requests.

Request format (JSON body for /generate):
  {
    "token_ids": [1, 2, 3, 4],   -- prompt as integer token IDs
    "max_new_tokens": 10          -- how many tokens to generate
  }

Response format:
  {
    "token_ids": [1, 2, 3, 4, 5, 6, ...]  -- full sequence (prompt + generated)
  }

The server is synchronous and single-threaded: one request is processed
at a time. A production server would run the scheduler in a background
thread and accept requests concurrently; this version keeps the threading
model simple so the scheduler's correctness -- not the concurrency model
-- is what's being demonstrated. The server's design is explicitly noted
as single-threaded in the README so the scope is honest.

Usage (with real weights, on a machine with Hub access):
  python server/server.py --checkpoint /path/to/llama-3.2-1b --port 8080

Usage (with random weights, for testing without Hub access):
  python server/server.py --random-weights --port 8080
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import LLAMA_3_2_1B, LlamaConfig
from scheduler.scheduler import Scheduler
from testutil.random_weights import random_model_weights


# Module-level scheduler instance, set in main() before the server starts.
# BaseHTTPRequestHandler is instantiated per-request by HTTPServer, so
# the scheduler must live outside the handler class.
_scheduler: Optional[Scheduler] = None
_config: Optional[LlamaConfig] = None


class InferenceHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Suppress the default per-request stdout logging so test output
        # stays clean. Remove this override if you want request logs.
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": f"unknown path: {self.path}"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": f"unknown path: {self.path}"})
            return

        # Read and parse request body.
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"invalid JSON: {e}"})
            return

        token_ids = req.get("token_ids")
        max_new_tokens = req.get("max_new_tokens")

        if not isinstance(token_ids, list) or not token_ids:
            self._send_json(400, {"error": "token_ids must be a non-empty list"})
            return
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            self._send_json(400, {"error": "max_new_tokens must be a positive integer"})
            return

        # Submit to scheduler and run until this request finishes.
        seq_id = _scheduler.add_request(token_ids, max_new_tokens)
        while _scheduler.get_result(seq_id) is None:
            _scheduler.step()

        result = _scheduler.get_result(seq_id)
        self._send_json(200, {"token_ids": result})

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_scheduler(
    config: LlamaConfig,
    weights,
    total_blocks: int = 512,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
) -> Scheduler:
    return Scheduler(
        config=config,
        weights=weights,
        total_blocks=total_blocks,
        block_size=block_size,
        dtype=dtype,
    )


def main():
    global _scheduler, _config

    parser = argparse.ArgumentParser(description="Llama inference server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a real Llama checkpoint directory")
    parser.add_argument("--random-weights", action="store_true",
                        help="Use random weights (for testing without a checkpoint)")
    parser.add_argument("--total-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    _config = LLAMA_3_2_1B

    if args.random_weights:
        print("Loading random weights (test mode)...")
        weights = random_model_weights(_config, num_layers=_config.num_hidden_layers)
    elif args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        from model.load_weights import load_model_weights
        weights = load_model_weights(args.checkpoint, _config)
    else:
        print("Error: provide --checkpoint or --random-weights", file=sys.stderr)
        sys.exit(1)

    _scheduler = make_scheduler(
        _config, weights,
        total_blocks=args.total_blocks,
        block_size=args.block_size,
    )

    server = HTTPServer(("127.0.0.1", args.port), InferenceHandler)
    print(f"Serving on http://127.0.0.1:{args.port}")
    print("Endpoints: POST /generate, GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
