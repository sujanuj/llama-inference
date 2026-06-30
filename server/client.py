"""A minimal test client for the inference server.

Sends a /generate request to a running server and prints the response.
Used for manual end-to-end testing once the server is running.

Usage (after starting server with --random-weights):
  python server/client.py --port 8080 --tokens 1 2 3 4 --max-new-tokens 8
"""

import argparse
import json
import sys
import urllib.request


def generate(host: str, port: int, token_ids: list, max_new_tokens: int) -> dict:
    url = f"http://{host}:{port}/generate"
    payload = json.dumps({"token_ids": token_ids, "max_new_tokens": max_new_tokens}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def health(host: str, port: int) -> dict:
    url = f"http://{host}:{port}/health"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Inference server test client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    print(f"Checking health at {args.host}:{args.port}...")
    h = health(args.host, args.port)
    print(f"Health: {h}")

    print(f"Sending generate request: token_ids={args.tokens}, max_new_tokens={args.max_new_tokens}")
    result = generate(args.host, args.port, args.tokens, args.max_new_tokens)
    print(f"Response: {result}")
    generated = result["token_ids"][len(args.tokens):]
    print(f"Generated tokens: {generated}")


if __name__ == "__main__":
    main()
