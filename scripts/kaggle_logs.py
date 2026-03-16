#!/usr/bin/env python
"""Fetch and print Kaggle kernel execution logs via the SDK API.

Usage:
    python scripts/kaggle_logs.py [kernel_slug] [--tail N] [--raw]

Defaults to mrkdagods/10-mtmc-pipeline-cityflowv2-full-run.
"""
import argparse
import re
import json
import sys


def fetch_logs(kernel: str, tail: int = 0, raw: bool = False) -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    if "/" not in kernel:
        print(f"ERROR: kernel must be owner/slug format, got: {kernel}", file=sys.stderr)
        sys.exit(1)

    owner, slug = kernel.split("/", 1)

    api = KaggleApi()
    api.authenticate()

    log_parts = []
    page_token = None
    with api.build_kaggle_client() as client:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner
            req.kernel_slug = slug
            req.page_size = 200
            if page_token:
                req.page_token = page_token
            resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
            if resp.log:
                log_parts.append(resp.log)
            if not resp.next_page_token:
                break
            page_token = resp.next_page_token

    raw_log = "".join(log_parts)

    if raw:
        return raw_log

    # Parse JSON-lines log format: [{"stream_name":..., "time":..., "data":...}, ...]
    # Strip wrapping [ ] and parse each entry
    clean_lines = []
    try:
        entries = json.loads(raw_log)
        for e in entries:
            data = e.get("data", "")
            # strip ANSI escape codes
            data = re.sub(r"\x1b\[[0-9;]*m", "", data)
            clean_lines.append(data)
    except (json.JSONDecodeError, TypeError):
        # fallback: treat as plain text, strip ANSI
        clean = re.sub(r"\x1b\[[0-9;]*m", "", raw_log)
        clean_lines = [clean]

    full = "".join(clean_lines)

    if tail > 0:
        lines = full.splitlines()
        full = "\n".join(lines[-tail:])

    return full


def main():
    parser = argparse.ArgumentParser(description="Fetch Kaggle kernel logs")
    parser.add_argument(
        "kernel",
        nargs="?",
        default="mrkdagods/10-mtmc-pipeline-cityflowv2-full-run",
        help="Kernel slug in owner/name format",
    )
    parser.add_argument("--tail", type=int, default=0, help="Show last N lines only")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON log without parsing")
    args = parser.parse_args()

    log = fetch_logs(args.kernel, tail=args.tail, raw=args.raw)
    print(log)


if __name__ == "__main__":
    main()
