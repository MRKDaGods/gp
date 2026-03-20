#!/usr/bin/env python
"""Fetch and print Kaggle kernel execution logs via the SDK API.

Usage:
    python scripts/kaggle_logs.py [kernel_slug] [--tail N] [--raw] [--wait]

Defaults to mrkdagods/10-mtmc-pipeline-cityflowv2-full-run.

Note: The Kaggle SDK only returns logs for *completed* kernel sessions.
While a kernel is RUNNING, use --wait to poll until it finishes and then
print the logs.
"""
import argparse
import re
import json
import sys
import time


def get_status(kernel: str) -> str:
    """Return kernel status string (e.g. 'RUNNING', 'COMPLETE', 'ERROR')."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest

    owner, slug = kernel.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        req = ApiGetKernelSessionStatusRequest()
        req.user_name = owner
        req.kernel_slug = slug
        resp = client.kernels.kernels_api_client.get_kernel_session_status(req)
        status = str(resp.status) if resp.status else ""
        # Also check failure message
        fail = resp.failure_message or ""
        return status, fail


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
    clean_lines = []
    try:
        entries = json.loads(raw_log)
        for e in entries:
            data = e.get("data", "")
            data = re.sub(r"\x1b\[[0-9;]*m", "", data)
            clean_lines.append(data)
    except (json.JSONDecodeError, TypeError):
        clean = re.sub(r"\x1b\[[0-9;]*m", "", raw_log)
        clean_lines = [clean]

    full = "".join(clean_lines)

    if tail > 0:
        lines = full.splitlines()
        full = "\n".join(lines[-tail:])

    return full


TERMINAL_STATUSES = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"}


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
    parser.add_argument("--out", default=None, help="Write output to this file instead of stdout")
    parser.add_argument(
        "--wait", action="store_true",
        help="If kernel is RUNNING, poll every --poll-interval seconds until done, then fetch logs",
    )
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between status polls (default: 60)")
    args = parser.parse_args()

    if "/" not in args.kernel:
        print(f"ERROR: kernel must be owner/slug format, got: {args.kernel}", file=sys.stderr)
        sys.exit(1)

    # Check status first
    status, fail_msg = get_status(args.kernel)
    # Normalize — strip enum prefix if present
    status_clean = status.replace("KernelWorkerStatus.", "").upper() if status else "UNKNOWN"

    if status_clean not in TERMINAL_STATUSES:
        if args.wait:
            print(f"Kernel is {status_clean} — waiting for completion (poll every {args.poll_interval}s)...",
                  file=sys.stderr)
            while status_clean not in TERMINAL_STATUSES:
                time.sleep(args.poll_interval)
                status, fail_msg = get_status(args.kernel)
                status_clean = status.replace("KernelWorkerStatus.", "").upper() if status else "UNKNOWN"
                ts = time.strftime("%H:%M:%S")
                print(f"  [{ts}] status: {status_clean}", file=sys.stderr)
            print(f"Kernel finished with status: {status_clean}", file=sys.stderr)
            if fail_msg:
                print(f"Failure message: {fail_msg}", file=sys.stderr)
        else:
            print(f"Kernel is {status_clean} — logs are only available after completion.", file=sys.stderr)
            print(f"Use --wait to poll until done, or check the Kaggle website for live logs.", file=sys.stderr)
            sys.exit(0)

    log = fetch_logs(args.kernel, tail=args.tail, raw=args.raw)

    if not log.strip():
        print(f"(No log output returned — kernel status: {status_clean})", file=sys.stderr)
        if fail_msg:
            print(f"Failure message: {fail_msg}", file=sys.stderr)
        sys.exit(0)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(log)
        print(f"Logs written to {args.out}")
    else:
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
        print(log)


if __name__ == "__main__":
    main()
