"""`athar` CLI entry point.

Commands land with their subsystems (ROADMAP phases). ``config resolve``
works today: it demonstrates the provenance contract — show exactly what a
run WOULD use and which layer set every value, before anything executes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _cmd_run(args: argparse.Namespace) -> int:
    import athar.components.adapters  # noqa: F401 — registers detector/tracker/embedders
    import athar.components.framesources  # noqa: F401 — registers frame sources
    from athar.contracts.authoring import parse_dotted_overrides
    from athar.contracts.config import ConfigLayer, ResolvedConfig
    from athar.contracts.manifest import RunManifest, RunRole
    from athar.contracts.store import FilesystemRunStore
    from athar.core.ids import new_run_id
    from athar.pipeline.ingest import IngestError, ingest_video
    from athar.pipeline.runner import PipelineRunner
    from athar.pipeline.stages.associate import AssociateStage
    from athar.pipeline.stages.detect_track import DetectTrackStage
    from athar.pipeline.stages.embed import EmbedStage
    from athar.pipeline.stages.index import IndexStage
    from athar.pipeline.stages.package import PackageStage
    from athar.profiles.builtin import ProfileError, load_profile

    try:
        profile, defaults = load_profile(args.profile)
    except ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    layers = [(ConfigLayer.PROFILE_DEFAULT, defaults)]
    if args.set:
        layers.append((ConfigLayer.RUN_OVERRIDE, parse_dotted_overrides(args.set)))
    config = ResolvedConfig.resolve(layers)

    store = FilesystemRunStore(args.runs_root)
    if args.resume:
        try:
            manifest = store.load(args.resume)
        except KeyError:
            print(f"error: run {args.resume!r} not found in {args.runs_root}", file=sys.stderr)
            return 2
        if manifest.profile_name != profile.name:
            print(
                f"error: run {manifest.run_id} was created with profile "
                f"{manifest.profile_name!r}, not {profile.name!r}",
                file=sys.stderr,
            )
            return 2
    else:
        if not args.video:
            print("error: at least one --video CAM=PATH is required", file=sys.stderr)
            return 2
        manifest = RunManifest(
            run_id=new_run_id(), role=RunRole(args.role), profile_name=profile.name
        )
        manifest.config = config
        for spec in args.video:
            cam, sep, path = spec.partition("=")
            if not sep or not cam or not path:
                print(f"error: --video expects CAM=PATH, got {spec!r}", file=sys.stderr)
                return 2
            timebase = None
            if Path(path).is_dir() and args.fps:
                from athar.core.timebase import CameraTimeBase, TimeBaseSource

                timebase = CameraTimeBase(
                    camera_id=cam, fps=args.fps, source=TimeBaseSource.MANUAL
                )
            try:
                video = ingest_video(manifest, cam, path, timebase=timebase)
            except IngestError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print(f"ingested {cam}: sha256={video.sha256[:12]}... fps={video.fps}")

    def console_sink(event) -> None:
        kind = getattr(event, "event", "")
        if kind == "stage_progress":
            cam = f" {event.camera_id}" if event.camera_id else ""
            print(f"\r  {event.stage}{cam}: {event.done}/{event.total}",
                  end="", file=sys.stderr)
        elif kind in ("stage_started", "stage_completed", "stage_skipped"):
            tail = {"stage_started": "...", "stage_completed": " done",
                    "stage_skipped": " already complete"}[kind]
            print(f"\n{event.stage}{tail}", end="", file=sys.stderr)

    stages = [DetectTrackStage(), EmbedStage(), IndexStage(), AssociateStage(),
              PackageStage()]
    runner = PipelineRunner(store, stages, extra_sinks=[console_sink])
    print(f"run {manifest.run_id} [{manifest.role.value}] profile={profile.name} "
          f"config={manifest.config.config_hash[:12]}")
    try:
        result = runner.run(manifest, profile)
    except Exception as exc:  # noqa: BLE001 — surfaced + recorded on the manifest
        print(f"\nrun FAILED: {exc}", file=sys.stderr)
        print(f"resume with: athar run --profile {args.profile} "
              f"--runs-root {args.runs_root} --resume {manifest.run_id}", file=sys.stderr)
        return 1
    print(f"\nrun {result.run_id}: {result.status.value}")
    print(f"artifacts: {len(result.artifacts)} "
          f"({', '.join(sorted(result.artifacts)[:8])}{'...' if len(result.artifacts) > 8 else ''})")
    return 0 if result.status.value == "completed" else 1


def _cmd_search(args: argparse.Namespace) -> int:
    from athar.contracts.store import FilesystemRunStore, RunNotFound
    from athar.search.engine import GallerySearcher, SearchError

    store = FilesystemRunStore(args.runs_root)
    try:
        gallery = store.load(args.gallery)
        probe = store.load(args.probe)
    except RunNotFound as exc:
        print(f"error: run not found: {exc}", file=sys.stderr)
        return 2
    try:
        searcher = GallerySearcher(store, gallery)
        stream = args.stream or next(
            (s for s in searcher.streams() if s != "hsv"), None
        )
        if stream is None:
            print("error: gallery has no appearance stream", file=sys.stderr)
            return 2
        hits = searcher.search_probe(
            store, probe, stream, top_k=args.top_k, min_score=args.min_score
        )
    except SearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not hits:
        print("no hits")
        return 0
    print(f"{len(hits)} hits on stream {stream!r} "
          f"(probe {probe.run_id} -> gallery {gallery.run_id}):")
    for hit in hits:
        print(
            f"  {hit.score:6.3f}  probe {hit.probe_key.camera_id}/{hit.probe_key.track_id}"
            f"  ->  {hit.gallery_key.camera_id}/{hit.gallery_key.track_id}"
            f"  [{hit.gallery_entity_class.value}] "
            f"t={hit.gallery_start_ts_s:.1f}-{hit.gallery_end_ts_s:.1f}s"
        )
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from athar.serving.lifecycle import (
        LifecycleError,
        ModelLifecycleDB,
        ModelNotFound,
        dump_entry,
        entry_summary,
        parse_metrics,
    )
    from athar.serving.registry import EvalReportRef, ModelStage, ModelTask

    db = ModelLifecycleDB(args.db)
    try:
        if args.subcommand == "list":
            task = ModelTask(args.task) if args.task else None
            stage = ModelStage(args.stage) if args.stage else None
            entries = db.list(task=task, stage=stage)
            if not entries:
                print("no models registered")
                return 0
            for entry in entries:
                print(entry_summary(entry))
        elif args.subcommand == "show":
            print(dump_entry(db.get(args.model_id)))
        elif args.subcommand == "import":
            result = db.import_yaml(args.yaml, actor=args.actor)
            print(f"added: {len(result['added'])} ({', '.join(result['added']) or '-'})")
            if result["skipped"]:
                print(f"skipped (already registered): {', '.join(result['skipped'])}")
        elif args.subcommand == "promote":
            report = None
            if args.eval_run:
                report = EvalReportRef(
                    run_id=args.eval_run,
                    benchmark=args.benchmark or "",
                    metrics=parse_metrics(args.metric or []),
                )
            entry = db.promote(
                args.model_id, ModelStage(args.to),
                eval_report=report, actor=args.actor, notes=args.notes,
            )
            print(f"{entry.model_id} -> {entry.stage.value}")
        elif args.subcommand == "retire":
            entry = db.retire(args.model_id, actor=args.actor)
            print(f"{entry.model_id} -> {entry.stage.value}")
        elif args.subcommand == "rollback":
            entry = db.rollback(ModelTask(args.task), actor=args.actor)
            print(f"production for {args.task} is now: "
                  f"{entry.model_id if entry.stage.value == 'production' else '(none)'}")
        elif args.subcommand == "events":
            for event in db.events(args.model_id):
                supersede = (
                    f" superseded={event['superseded_model']}"
                    if event["superseded_model"] else ""
                )
                print(f"{event['ts']}  {event['model_id']}: {event['action']} "
                      f"{event['from_stage'] or '-'} -> {event['to_stage'] or '-'}"
                      f"{supersede} [{event['actor'] or 'unattributed'}]")
    except (LifecycleError, ModelNotFound, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        db.close()
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

    run = sub.add_parser("run", help="execute the pipeline on evidence videos")
    run.add_argument("--profile", default="multiclass",
                     help="builtin profile name or profile YAML path")
    run.add_argument("--video", action="append", metavar="CAM=PATH",
                     help="evidence video (or image-sequence directory) for one "
                          "camera (repeatable)")
    run.add_argument("--fps", type=float, default=None,
                     help="declared fps for image-directory cameras (they have "
                          "no intrinsic rate; recorded as a MANUAL TimeBase)")
    run.add_argument("--role", default="gallery",
                     choices=["gallery", "probe", "benchmark", "adaptation"])
    run.add_argument("--runs-root", default="data/runs", help="run store root")
    run.add_argument("--resume", metavar="RUN_ID",
                     help="resume an interrupted run instead of creating one")
    run.add_argument("--set", action="append", metavar="KEY.PATH=VALUE",
                     help="run-level config override")
    run.set_defaults(handler=_cmd_run)

    search = sub.add_parser("search", help="search a gallery run with a probe run")
    search.add_argument("--gallery", required=True, metavar="RUN_ID")
    search.add_argument("--probe", required=True, metavar="RUN_ID")
    search.add_argument("--stream", help="embedding stream (default: first appearance stream)")
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--min-score", type=float, default=0.0)
    search.add_argument("--runs-root", default="data/runs")
    search.set_defaults(handler=_cmd_search)
    models = sub.add_parser("models", help="model lifecycle registry (D5)")
    models_sub = models.add_subparsers(dest="subcommand", required=True)

    def _models_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", default="data/registry/models.db",
                       help="lifecycle registry SQLite path")
        p.add_argument("--actor", default="", help="who is performing this action")
        p.set_defaults(handler=_cmd_models)

    m_list = models_sub.add_parser("list", help="list registered models")
    m_list.add_argument("--task", help="filter by task (e.g. reid_vehicle)")
    m_list.add_argument("--stage", help="filter by stage (candidate/validated/production/retired)")
    _models_common(m_list)
    m_show = models_sub.add_parser("show", help="dump one model entry as JSON")
    m_show.add_argument("model_id")
    _models_common(m_show)
    m_import = models_sub.add_parser(
        "import", help="import authoring YAML (new models enter as candidates)"
    )
    m_import.add_argument("yaml")
    _models_common(m_import)
    m_promote = models_sub.add_parser(
        "promote", help="eval-gated promotion (candidate->validated->production)"
    )
    m_promote.add_argument("model_id")
    m_promote.add_argument("--to", required=True, choices=["validated", "production"])
    m_promote.add_argument("--eval-run", help="evaluation run id backing this promotion")
    m_promote.add_argument("--benchmark", help="benchmark the eval ran on")
    m_promote.add_argument("--metric", action="append", metavar="NAME=VALUE",
                           help="eval metric (repeatable)")
    m_promote.add_argument("--notes", default="")
    _models_common(m_promote)
    m_retire = models_sub.add_parser("retire", help="retire a model (any stage)")
    m_retire.add_argument("model_id")
    _models_common(m_retire)
    m_rollback = models_sub.add_parser(
        "rollback", help="undo the latest production promotion for a task"
    )
    m_rollback.add_argument("--task", required=True)
    _models_common(m_rollback)
    m_events = models_sub.add_parser("events", help="lifecycle audit trail")
    m_events.add_argument("model_id", nargs="?", default=None)
    _models_common(m_events)
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
