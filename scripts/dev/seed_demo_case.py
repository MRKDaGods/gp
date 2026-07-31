"""Seed the MIE finals demo case through the real HTTP API.

Creates "El Shorouk — Finals Demo": attaches the gallery + probe runs,
runs a probe->gallery search for ONE person and ONE vehicle stream,
creates a target for each, proposes the top hits as hypotheses, and
confirms the strongest hit of each class — so the workspace opens with a
verified cross-camera match backed by an audit trail.

Everything goes through the API (never the DB): the same audited code
paths an investigator uses, so the audit chain tells a true story.

Usage:
  python scripts/dev/seed_demo_case.py --gallery RUN_ID --probe RUN_ID \
      [--api http://127.0.0.1:8000] [--user demo --password demo-pass-1]

Idempotence: if a case with the same title already exists (visible to
this user), the script aborts rather than duplicating — delete the old
case first or pass --title to create a differently named one.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_TITLE = "El Shorouk — Finals Demo"


class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.cookie: str | None = None

    def call(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(self.base + path, method=method)
        req.add_header("Content-Type", "application/json")
        if self.cookie:
            req.add_header("Cookie", self.cookie)
        data = json.dumps(body).encode() if body is not None else None
        try:
            with urllib.request.urlopen(req, data=data) as resp:
                set_cookie = resp.headers.get("Set-Cookie")
                if set_cookie:
                    self.cookie = set_cookie.split(";", 1)[0]
                payload = resp.read()
                return json.loads(payload) if payload else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"{method} {path} -> {exc.code}: {detail}") from None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--user", default="demo")
    parser.add_argument("--password", default="demo-pass-1")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vehicle-stream", default="transreid_primary")
    parser.add_argument("--person-stream", default="transreid_person")
    args = parser.parse_args()

    api = Api(args.api)
    me = api.call("POST", "/auth/login",
                  {"username": args.user, "password": args.password})
    print(f"logged in as {me['username']} [{me['role']}]")

    for case in api.call("GET", "/cases"):
        if case.get("title") == args.title:
            print(f"case {case['case_id']} already titled {args.title!r} — aborting "
                  "(delete it or pass --title)", file=sys.stderr)
            return 2

    case = api.call("POST", "/cases", {"title": args.title})
    case_id = case["case_id"]
    print(f"case: {case_id}  {args.title!r}")
    for run_id in (args.gallery, args.probe):
        api.call("POST", f"/cases/{case_id}/runs", {"run_id": run_id})
        print(f"attached run {run_id}")

    seeded = []
    for label, stream, target_label in [
        ("vehicle", args.vehicle_stream, "Vehicle of interest"),
        ("person", args.person_stream, "Person of interest"),
    ]:
        result = api.call("POST", "/search", {
            "gallery_run_id": args.gallery,
            "probe_run_id": args.probe,
            "stream": stream,
            "top_k": args.top_k,
        })
        hits = result["hits"]
        if not hits:
            print(f"{label}: NO HITS on {stream} — skipping", file=sys.stderr)
            continue
        # top_k limits hits PER PROBE TRACKLET; a real case files hypotheses
        # for ONE suspect, so keep only the best probe tracklet's hits
        best = hits[0]
        suspect = (best["probe_camera_id"], best["probe_track_id"])
        hits = [
            h for h in hits
            if (h["probe_camera_id"], h["probe_track_id"]) == suspect
        ][: args.top_k]
        print(f"{label}: {len(hits)} hits on {stream}; best "
              f"{best['probe_camera_id']}/{best['probe_track_id']} -> "
              f"{best['gallery_camera_id']}/{best['gallery_track_id']} "
              f"score={best['score']:.3f} p={best.get('probability')}")

        target = api.call("POST", f"/cases/{case_id}/targets",
                          {"label": target_label})
        target_id = target["target_id"]
        first_hyp_id = None
        for hit in hits:
            hyp = api.call(
                "POST", f"/cases/{case_id}/targets/{target_id}/hypotheses",
                {
                    "kind": "appearance",
                    "run_id": args.gallery,
                    "camera_id": hit["gallery_camera_id"],
                    "track_id": hit["gallery_track_id"],
                    "raw_score": hit["score"],
                    "probability": hit.get("probability"),
                    "stream": hit["stream"],
                },
            )
            if first_hyp_id is None:
                first_hyp_id = hyp["hypothesis_id"]
        confirmed = api.call(
            "POST",
            f"/cases/{case_id}/targets/{target_id}/hypotheses/{first_hyp_id}/decide",
            {"status": "confirmed"},
        )
        print(f"{label}: target {target_id}, {len(hits)} hypotheses, "
              f"top hit CONFIRMED (status={confirmed['status']})")
        seeded.append(label)

    if len(seeded) < 2:
        print("WARNING: demo case is missing a class — inspect search results",
              file=sys.stderr)
        return 1
    print(f"\ndemo case ready: /ar/cases/{case_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
