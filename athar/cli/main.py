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


def _parse_video_specs(specs: list[str] | None) -> dict[str, str]:
    """CAM=PATH pairs -> dict; raises ValueError on malformed specs."""
    videos: dict[str, str] = {}
    for spec in specs or []:
        cam, sep, path = spec.partition("=")
        if not sep or not cam or not path:
            raise ValueError(f"--video expects CAM=PATH, got {spec!r}")
        videos[cam] = path
    return videos


def _cmd_run(args: argparse.Namespace) -> int:
    from athar.contracts.authoring import ConfigAuthoringError
    from athar.contracts.store import FilesystemRunStore, RunNotFound
    from athar.pipeline.ingest import IngestError
    from athar.pipeline.runner import PipelineRunner
    from athar.pipeline.setup import (
        RunSetupError,
        create_run,
        default_stages,
        resume_run,
    )
    from athar.profiles.builtin import ProfileError

    store = FilesystemRunStore(args.runs_root)
    try:
        if args.resume:
            try:
                manifest, profile = resume_run(store, args.resume, args.profile)
            except RunNotFound:
                print(f"error: run {args.resume!r} not found in {args.runs_root}",
                      file=sys.stderr)
                return 2
        else:
            if not args.video:
                print("error: at least one --video CAM=PATH is required", file=sys.stderr)
                return 2
            manifest, profile = create_run(
                profile_name=args.profile,
                videos=_parse_video_specs(args.video),
                role=args.role,
                fps=args.fps,
                overrides=args.set,
                on_ingest=lambda cam, video: print(
                    f"ingested {cam}: sha256={video.sha256[:12]}... fps={video.fps}"
                ),
            )
    except (ProfileError, IngestError, RunSetupError, ConfigAuthoringError,
            ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

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

    runner = PipelineRunner(store, default_stages(), extra_sinks=[console_sink])
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


def _cmd_worker(args: argparse.Namespace) -> int:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    from athar.jobs.worker import run_worker

    return run_worker(
        args.queue, once=args.once, poll_s=args.poll, worker_id=args.worker_id
    )


def _cmd_jobs(args: argparse.Namespace) -> int:
    from athar.jobs.queue import JobExecutor, JobNotFound, JobQueue, JobStatus
    from athar.jobs.service import JobService

    try:
        if args.subcommand == "submit":
            try:
                videos = _parse_video_specs(args.video)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            service = JobService(args.queue, args.runs_root, spawn_worker=False)
            try:
                job = service.submit_run(
                    videos=videos,
                    profile=args.profile,
                    role=args.role,
                    fps=args.fps,
                    overrides=args.set,
                    resume_run_id=args.resume,
                    executor=JobExecutor(args.executor),
                    priority=args.priority,
                )
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            finally:
                service.queue.close()
            print(job.job_id)
            print("start a worker with: athar worker --queue " + args.queue,
                  file=sys.stderr)
            return 0

        queue = JobQueue(args.queue)
        try:
            if args.subcommand == "list":
                status = JobStatus(args.status) if args.status else None
                jobs = queue.list(status=status)
                if not jobs:
                    print("no jobs")
                for job in jobs:
                    run = f" run={job.run_id}" if job.run_id else ""
                    err = f" error={job.error}" if job.error else ""
                    print(f"{job.job_id}  {job.kind:<14} {job.status.value:<10} "
                          f"[{job.executor.value}]{run}{err}")
            elif args.subcommand == "show":
                print(queue.get(args.job_id).model_dump_json(indent=2))
            elif args.subcommand == "cancel":
                status = queue.request_cancel(args.job_id)
                print(f"{args.job_id}: {status.value}")
        finally:
            queue.close()
    except (JobNotFound, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_users(args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from athar.api import audit
    from athar.api.db import Role, UserRow, make_engine, make_session_factory
    from athar.api.security import AuthError, create_user

    engine = make_engine(args.app_db)
    factory = make_session_factory(engine)
    try:
        with factory() as db:
            if args.subcommand == "add":
                try:
                    user = create_user(
                        db, args.username, args.password, Role(args.role)
                    )
                except AuthError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2
                audit.append(db, "cli", "user_created",
                             username=user.username, role=user.role)
                db.commit()
                print(f"created {user.username} [{user.role}]")
            elif args.subcommand == "list":
                users = db.scalars(select(UserRow).order_by(UserRow.username)).all()
                if not users:
                    print("no users (create one with: athar users add ...)")
                for user in users:
                    flag = " (disabled)" if user.disabled else ""
                    print(f"{user.username:<24} {user.role}{flag}")
    finally:
        engine.dispose()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from athar.api.app import create_app
    from athar.api.settings import ApiSettings

    app = create_app(ApiSettings())  # paths/flags come from ATHAR_* env vars
    uvicorn.run(app, host=args.host, port=args.port)
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
    users = sub.add_parser("users", help="manage API users (local operator only)")
    users_sub = users.add_subparsers(dest="subcommand", required=True)
    u_add = users_sub.add_parser("add", help="create a user")
    u_add.add_argument("username")
    u_add.add_argument("--password", required=True)
    u_add.add_argument("--role", default="viewer",
                       choices=["viewer", "investigator", "admin"])
    u_add.add_argument("--app-db", default="data/app/app.db")
    u_add.set_defaults(handler=_cmd_users)
    u_list = users_sub.add_parser("list", help="list users")
    u_list.add_argument("--app-db", default="data/app/app.db")
    u_list.set_defaults(handler=_cmd_users)

    serve = sub.add_parser("serve", help="run the API server (settings via ATHAR_* env)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_cmd_serve)

    worker = sub.add_parser("worker", help="run a job-queue worker process")
    worker.add_argument("--queue", default="data/jobs/jobs.db", help="job queue SQLite path")
    worker.add_argument("--once", action="store_true",
                        help="process at most one job, then exit")
    worker.add_argument("--poll", type=float, default=1.0,
                        help="idle poll interval in seconds")
    worker.add_argument("--worker-id", default=None)
    worker.set_defaults(handler=_cmd_worker)

    jobs = sub.add_parser("jobs", help="submit and inspect queued jobs")
    jobs_sub = jobs.add_subparsers(dest="subcommand", required=True)
    j_submit = jobs_sub.add_parser("submit", help="queue a pipeline run")
    j_submit.add_argument("--queue", default="data/jobs/jobs.db")
    j_submit.add_argument("--runs-root", default="data/runs")
    j_submit.add_argument("--profile", default="multiclass")
    j_submit.add_argument("--video", action="append", metavar="CAM=PATH")
    j_submit.add_argument("--role", default="gallery",
                          choices=["gallery", "probe", "benchmark", "adaptation"])
    j_submit.add_argument("--fps", type=float, default=None)
    j_submit.add_argument("--set", action="append", metavar="KEY.PATH=VALUE")
    j_submit.add_argument("--resume", metavar="RUN_ID", default=None)
    j_submit.add_argument("--executor", default="local", choices=["local", "kaggle"])
    j_submit.add_argument("--priority", type=int, default=0)
    j_submit.set_defaults(handler=_cmd_jobs)
    for name, help_text in [("list", "list jobs"), ("show", "dump one job"),
                            ("cancel", "request cancellation")]:
        p = jobs_sub.add_parser(name, help=help_text)
        p.add_argument("--queue", default="data/jobs/jobs.db")
        if name == "list":
            p.add_argument("--status", default=None)
        else:
            p.add_argument("job_id")
        p.set_defaults(handler=_cmd_jobs)

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
