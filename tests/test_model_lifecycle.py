"""Model lifecycle registry tests: eval-gated promotion, demote-on-promote,
rollback, YAML authoring rules, event trail, persistence."""

from __future__ import annotations

import pytest

from athar.serving.lifecycle import (
    LifecycleError,
    ModelLifecycleDB,
    ModelNotFound,
    parse_metrics,
)
from athar.serving.registry import (
    CheckpointRef,
    EvalReportRef,
    ModelEntry,
    ModelStage,
    ModelTask,
)


def make_entry(model_id: str, task: ModelTask = ModelTask.REID_VEHICLE) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        task=task,
        architecture="transreid_vit_base",
        checkpoint=CheckpointRef(
            sha256="ab" * 32, size_bytes=123456, filename=f"{model_id}.pth"
        ),
    )


def eval_ref(run_id: str = "run-eval-1") -> EvalReportRef:
    return EvalReportRef(run_id=run_id, benchmark="veri776", metrics={"mAP": 0.93})


@pytest.fixture()
def db(tmp_path):
    handle = ModelLifecycleDB(tmp_path / "models.db")
    yield handle
    handle.close()


class TestCrud:
    def test_add_get_roundtrip(self, db):
        entry = make_entry("m1")
        db.add(entry, actor="tester")
        loaded = db.get("m1")
        assert loaded == entry
        assert loaded.stage is ModelStage.CANDIDATE

    def test_duplicate_add_refused(self, db):
        db.add(make_entry("m1"))
        with pytest.raises(LifecycleError, match="already registered"):
            db.add(make_entry("m1"))

    def test_get_missing_raises(self, db):
        with pytest.raises(ModelNotFound):
            db.get("ghost")

    def test_list_filters(self, db):
        db.add(make_entry("v1", ModelTask.REID_VEHICLE))
        db.add(make_entry("p1", ModelTask.REID_PERSON))
        assert [e.model_id for e in db.list()] == ["p1", "v1"]
        assert [e.model_id for e in db.list(task=ModelTask.REID_PERSON)] == ["p1"]
        assert db.list(stage=ModelStage.PRODUCTION) == []

    def test_persists_across_reopen(self, db, tmp_path):
        db.add(make_entry("m1"))
        db.close()
        reopened = ModelLifecycleDB(tmp_path / "models.db")
        try:
            assert reopened.get("m1").model_id == "m1"
        finally:
            reopened.close()


class TestPromotion:
    def test_promotion_requires_eval_report(self, db):
        db.add(make_entry("m1"))
        with pytest.raises(ValueError, match="evaluation report"):
            db.promote("m1", ModelStage.VALIDATED)
        assert db.get("m1").stage is ModelStage.CANDIDATE  # unchanged

    def test_skipping_stages_refused(self, db):
        db.add(make_entry("m1"))
        with pytest.raises(ValueError, match="illegal promotion"):
            db.promote("m1", ModelStage.PRODUCTION, eval_report=eval_ref())

    def test_full_ladder(self, db):
        db.add(make_entry("m1"))
        db.promote("m1", ModelStage.VALIDATED, eval_report=eval_ref("e1"))
        db.promote("m1", ModelStage.PRODUCTION, eval_report=eval_ref("e2"))
        entry = db.get("m1")
        assert entry.stage is ModelStage.PRODUCTION
        assert [r.run_id for r in entry.eval_reports] == ["e1", "e2"]
        assert db.production(ModelTask.REID_VEHICLE).model_id == "m1"

    def test_promote_demotes_previous_production(self, db):
        for mid in ("old", "new"):
            db.add(make_entry(mid))
            db.promote(mid, ModelStage.VALIDATED, eval_report=eval_ref())
        db.promote("old", ModelStage.PRODUCTION, eval_report=eval_ref())
        db.promote("new", ModelStage.PRODUCTION, eval_report=eval_ref())
        assert db.get("old").stage is ModelStage.VALIDATED
        assert db.production(ModelTask.REID_VEHICLE).model_id == "new"

    def test_other_tasks_unaffected(self, db):
        db.add(make_entry("veh"))
        db.add(make_entry("per", ModelTask.REID_PERSON))
        for mid in ("veh", "per"):
            db.promote(mid, ModelStage.VALIDATED, eval_report=eval_ref())
            db.promote(mid, ModelStage.PRODUCTION, eval_report=eval_ref())
        assert db.production(ModelTask.REID_VEHICLE).model_id == "veh"
        assert db.production(ModelTask.REID_PERSON).model_id == "per"

    def test_retire_from_any_stage(self, db):
        db.add(make_entry("m1"))
        db.retire("m1")
        assert db.get("m1").stage is ModelStage.RETIRED


class TestRollback:
    def test_rollback_restores_superseded(self, db):
        for mid in ("old", "new"):
            db.add(make_entry(mid))
            db.promote(mid, ModelStage.VALIDATED, eval_report=eval_ref())
        db.promote("old", ModelStage.PRODUCTION, eval_report=eval_ref())
        db.promote("new", ModelStage.PRODUCTION, eval_report=eval_ref())
        restored = db.rollback(ModelTask.REID_VEHICLE, actor="ops")
        assert restored.model_id == "old"
        assert db.production(ModelTask.REID_VEHICLE).model_id == "old"
        assert db.get("new").stage is ModelStage.VALIDATED

    def test_rollback_without_predecessor_leaves_no_production(self, db):
        db.add(make_entry("only"))
        db.promote("only", ModelStage.VALIDATED, eval_report=eval_ref())
        db.promote("only", ModelStage.PRODUCTION, eval_report=eval_ref())
        result = db.rollback(ModelTask.REID_VEHICLE)
        assert result.model_id == "only"
        assert result.stage is ModelStage.VALIDATED
        assert db.production(ModelTask.REID_VEHICLE) is None

    def test_rollback_without_production_refused(self, db):
        with pytest.raises(LifecycleError, match="no production model"):
            db.rollback(ModelTask.REID_VEHICLE)

    def test_rollback_skips_retired_predecessor(self, db):
        for mid in ("old", "new"):
            db.add(make_entry(mid))
            db.promote(mid, ModelStage.VALIDATED, eval_report=eval_ref())
        db.promote("old", ModelStage.PRODUCTION, eval_report=eval_ref())
        db.promote("new", ModelStage.PRODUCTION, eval_report=eval_ref())
        db.retire("old")
        result = db.rollback(ModelTask.REID_VEHICLE)
        assert result.stage is ModelStage.VALIDATED  # "new" demoted, none restored
        assert db.production(ModelTask.REID_VEHICLE) is None
        assert db.get("old").stage is ModelStage.RETIRED  # untouched


class TestEvents:
    def test_trail_records_transitions(self, db):
        db.add(make_entry("m1"), actor="alice")
        db.promote("m1", ModelStage.VALIDATED, eval_report=eval_ref("e1"), actor="bob")
        actions = [(e["action"], e["actor"]) for e in db.events("m1")]
        assert actions == [("register", "alice"), ("promote", "bob")]
        promote = db.events("m1")[-1]
        assert promote["eval_run_id"] == "e1"
        assert promote["from_stage"] == "candidate"
        assert promote["to_stage"] == "validated"


class TestAuthoring:
    def test_import_yaml_creates_candidates(self, db, tmp_path):
        path = tmp_path / "authoring.yaml"
        entry = make_entry("m1")
        path.write_text(
            "models:\n"
            f"  - model_id: m1\n"
            f"    task: {entry.task.value}\n"
            f"    architecture: {entry.architecture}\n"
            f"    checkpoint:\n"
            f"      sha256: {entry.checkpoint.sha256}\n"
            f"      size_bytes: {entry.checkpoint.size_bytes}\n"
            f"      filename: {entry.checkpoint.filename}\n",
            encoding="utf-8",
        )
        result = db.import_yaml(path, actor="ci")
        assert result == {"added": ["m1"], "skipped": []}
        assert db.get("m1").stage is ModelStage.CANDIDATE
        # idempotent re-import
        assert db.import_yaml(path) == {"added": [], "skipped": ["m1"]}

    def test_import_refuses_authored_production(self, db, tmp_path):
        path = tmp_path / "sneaky.yaml"
        path.write_text(
            "models:\n"
            "  - model_id: sneaky\n"
            "    task: reid_vehicle\n"
            "    architecture: x\n"
            "    stage: production\n"
            "    checkpoint: {sha256: 'ab', size_bytes: 1, filename: x.pth}\n",
            encoding="utf-8",
        )
        with pytest.raises(LifecycleError, match="only introduce candidates"):
            db.import_yaml(path)

    def test_import_requires_models_list(self, db, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("nothing: here\n", encoding="utf-8")
        with pytest.raises(LifecycleError, match="'models' list"):
            db.import_yaml(path)


class TestCliHelpers:
    def test_parse_metrics(self):
        assert parse_metrics(["mAP=0.93", "rank1=0.98"]) == {"mAP": 0.93, "rank1": 0.98}

    def test_parse_metrics_rejects_garbage(self):
        with pytest.raises(LifecycleError):
            parse_metrics(["mAP"])
        with pytest.raises(LifecycleError):
            parse_metrics(["mAP=high"])


class TestCli:
    def test_import_promote_list_flow(self, tmp_path, capsys):
        from athar.cli.main import main

        db_path = str(tmp_path / "models.db")
        yaml_path = tmp_path / "authoring.yaml"
        yaml_path.write_text(
            "models:\n"
            "  - model_id: cli_model\n"
            "    task: reid_person\n"
            "    architecture: transreid\n"
            "    checkpoint: {sha256: 'cd', size_bytes: 2, filename: p.pth}\n",
            encoding="utf-8",
        )
        assert main(["models", "import", str(yaml_path), "--db", db_path]) == 0
        assert main([
            "models", "promote", "cli_model", "--to", "validated",
            "--eval-run", "run-1", "--benchmark", "market1501",
            "--metric", "mAP=0.8", "--db", db_path, "--actor", "cli",
        ]) == 0
        assert main(["models", "list", "--db", db_path]) == 0
        out = capsys.readouterr().out
        assert "cli_model -> validated" in out
        assert "reid_person" in out

    def test_promote_without_eval_fails(self, tmp_path, capsys):
        from athar.cli.main import main

        db_path = str(tmp_path / "models.db")
        yaml_path = tmp_path / "authoring.yaml"
        yaml_path.write_text(
            "models:\n"
            "  - model_id: m\n"
            "    task: reid_person\n"
            "    architecture: a\n"
            "    checkpoint: {sha256: 'cd', size_bytes: 2, filename: p.pth}\n",
            encoding="utf-8",
        )
        main(["models", "import", str(yaml_path), "--db", db_path])
        rc = main(["models", "promote", "m", "--to", "validated", "--db", db_path])
        assert rc == 2
        assert "evaluation report" in capsys.readouterr().err
