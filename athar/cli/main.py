"""`athar` CLI entry point.

Commands land with their subsystems (ROADMAP phases). ``config resolve``
works today: it demonstrates the provenance contract — show exactly what a
run WOULD use and which layer set every value, before anything executes.
"""

from __future__ import annotations

import argparse
import sys

from athar import __version__


def _cmd_config_resolve(args: argparse.Namespace) -> int:
    from athar.contracts.authoring import ConfigAuthoringError, resolve_from_files

    try:
        cfg = resolve_from_files(
            profile_defaults=args.profile,
            deployment=args.deployment,
            case=args.case,
            overrides=args.set or [],
        )
    except ConfigAuthoringError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    width = max((len(k) for k in cfg.values), default=0)
    for key in sorted(cfg.values):
        layer = cfg.provenance[key].value
        print(f"{key.ljust(width)}  = {cfg.values[key]!r:<24} [{layer}]")
    print(f"\nconfig_hash: {cfg.config_hash}")
    return 0


def _not_implemented(what: str, phase: str):
    def handler(_args: argparse.Namespace) -> int:
        print(f"`athar {what}` arrives in {phase} — see ROADMAP.md", file=sys.stderr)
        return 3

    return handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="athar", description=__doc__)
    parser.add_argument("--version", action="version", version=f"athar {__version__}")
    sub = parser.add_subparsers(dest="command")

    config = sub.add_parser("config", help="configuration tools")
    config_sub = config.add_subparsers(dest="subcommand", required=True)
    resolve = config_sub.add_parser(
        "resolve", help="resolve config layers and show per-key provenance"
    )
    resolve.add_argument("--profile", required=True, help="profile defaults YAML")
    resolve.add_argument("--deployment", help="deployment layer YAML")
    resolve.add_argument("--case", help="case layer YAML")
    resolve.add_argument(
        "--set", action="append", metavar="KEY.PATH=VALUE", help="run-level override"
    )
    resolve.set_defaults(handler=_cmd_config_resolve)

    run = sub.add_parser("run", help="submit/execute a pipeline run")
    run.set_defaults(handler=_not_implemented("run", "Phase 2"))
    models = sub.add_parser("models", help="model lifecycle registry")
    models.set_defaults(handler=_not_implemented("models", "Phase 4"))
    migrate = sub.add_parser("migrate", help="convert v1 run dirs to v2 manifests")
    migrate.set_defaults(handler=_not_implemented("migrate", "Phase 5"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
