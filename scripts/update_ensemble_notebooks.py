import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_10A = ROOT / "notebooks" / "kaggle" / "10a_stages012" / "mtmc-10a-stages-0-2-tracking-reid-features.ipynb"
NOTEBOOK_10C = ROOT / "notebooks" / "kaggle" / "10c_stages45" / "mtmc-10c-stages-4-5-association-eval.ipynb"


def to_source(lines: list[str]) -> list[str]:
    if not lines:
        return []
    return [f"{line}\n" for line in lines[:-1]] + [lines[-1]]


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_notebook(path: Path, notebook: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, ensure_ascii=True, indent=1)
        handle.write("\n")


def replace_line(lines: list[str], old: str, new: str) -> bool:
    for index, line in enumerate(lines):
        if line == old:
            lines[index] = new
            return True
    return False


def ensure_line(lines: list[str], old: str, new: str) -> str:
    if replace_line(lines, old, new):
        return "updated"
    if new in lines:
        return "unchanged"
    raise RuntimeError(f"Expected one of these lines to exist: {old} | {new}")


def ensure_one_of(lines: list[str], old_lines: list[str], new: str) -> str:
    for old in old_lines:
        if replace_line(lines, old, new):
            return "updated"
    if new in lines:
        return "unchanged"
    joined_old = " | ".join(old_lines)
    raise RuntimeError(f"Expected one of these lines to exist: {joined_old} | {new}")


def insert_after(lines: list[str], anchor: str, new_lines: list[str]) -> bool:
    for index, line in enumerate(lines):
        if line == anchor:
            existing_slice = lines[index + 1:index + 1 + len(new_lines)]
            if existing_slice != new_lines:
                lines[index + 1:index + 1] = new_lines
            return True
    return False


def remove_line(lines: list[str], target: str) -> bool:
    for index, line in enumerate(lines):
        if line == target:
            del lines[index]
            return True
    return False


def update_10a() -> list[str]:
    notebook = load_notebook(NOTEBOOK_10A)
    changes: list[str] = []

    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        source = cell.get("source", [])
        joined = "".join(source)
        if 'stage2.reid.vehicle2.enabled=false' not in joined and 'stage2.reid.vehicle2.enabled=true' not in joined:
            continue

        lines = joined.splitlines()
        status = ensure_line(
            lines,
            '    "--override", "stage2.reid.vehicle2.enabled=false",',
            '    "--override", "stage2.reid.vehicle2.enabled=true",',
        )

        cell["source"] = to_source(lines)
        save_notebook(NOTEBOOK_10A, notebook)
        reloaded = load_notebook(NOTEBOOK_10A)
        _ = reloaded["cells"][cell_index - 1]["source"]
        if status == "updated":
            changes.append(
                f"10a cell {cell_index}: stage2.reid.vehicle2.enabled=false -> stage2.reid.vehicle2.enabled=true"
            )
        else:
            changes.append(
                f"10a cell {cell_index}: stage2.reid.vehicle2.enabled already true"
            )
        return changes

    raise RuntimeError("10a vehicle2 override cell not found")


def update_10c() -> list[str]:
    notebook = load_notebook(NOTEBOOK_10C)
    changes: list[str] = []
    params_index: int | None = None
    run_index: int | None = None
    misplaced_run_markdown_index: int | None = None
    misplaced_run_lines: list[str] | None = None

    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        joined = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and 'print("Stage 4 params' in joined and "MTMC_ONLY = False" in joined and "FUSION_WEIGHT" in joined:
            params_index = cell_index - 1

        if '"scripts/run_pipeline.py"' in joined and '"--stages", "4,5"' in joined and 'WARNING: GT_DIR is empty' in joined:
            if cell.get("cell_type") == "code":
                run_index = cell_index - 1
            elif cell.get("cell_type") == "markdown":
                misplaced_run_markdown_index = cell_index - 1
                misplaced_run_lines = joined.splitlines()

    if params_index is None:
        raise RuntimeError("10c params cell not found")

    params_cell = notebook["cells"][params_index]
    params_lines = "".join(params_cell.get("source", [])).splitlines()
    param_replacements = [
        ([
            '# v46 optimized config (local best: IDF1=0.8297)',
            '# v75: CONSOLIDATED OPTIMAL — all params from v62-v73 sweeps',
            '# v80 best + ensemble configuration',
        ], '# v80 best + ensemble configuration'),
        ([
            'AQE_K             = 2     # v46: 2 (optimal after QE self-exclusion fix)',
            'AQE_K             = 3     # v62: 3 optimal (+0.9pp over 2)',
            'AQE_K             = 3',
        ], 'AQE_K             = 3'),
        ([
            'ALGORITHM         = "connected_components"',
            'ALGORITHM         = "conflict_free_cc"  # v67: +0.21pp over CC',
            'ALGORITHM         = "conflict_free_cc"',
        ], 'ALGORITHM         = "conflict_free_cc"'),
        ([
            'APPEARANCE_WEIGHT = 0.75  # v46: CityFlowV2 vehicle config',
            'APPEARANCE_WEIGHT = 0.70  # v73: +0.76pp over 0.75',
            'APPEARANCE_WEIGHT = 0.70',
        ], 'APPEARANCE_WEIGHT = 0.70'),
        ([
            'INTRA_MERGE_THRESH = 0.75',
            'INTRA_MERGE_THRESH = 0.80  # v72: +0.14pp over 0.75',
            'INTRA_MERGE_THRESH = 0.80',
        ], 'INTRA_MERGE_THRESH = 0.80'),
        ([
            'INTRA_MERGE_GAP   = 60    # seconds',
            'INTRA_MERGE_GAP   = 30     # v72: gap-insensitive at 0.80',
            'INTRA_MERGE_GAP   = 30',
        ], 'INTRA_MERGE_GAP   = 30'),
        ([
            'FUSION_WEIGHT     = 0.0    # Phase 1: no secondary model',
            'FUSION_WEIGHT     = 0.1   # v53: 10% secondary (was 0.3 in v52)',
            'FUSION_WEIGHT     = 0.3',
        ], 'FUSION_WEIGHT     = 0.3'),
        (['CAMERA_BIAS       = True', 'CAMERA_BIAS       = False'], 'CAMERA_BIAS       = False'),
        (['ZONE_MODEL        = True', 'ZONE_MODEL        = False'], 'ZONE_MODEL        = False'),
        ([
            'print("Stage 4 params (v46 optimized):")',
            'print("Stage 4 params (v75 consolidated optimal):")',
            'print("Stage 4 params (v80 best + ensemble):")',
        ], 'print("Stage 4 params (v80 best + ensemble):")'),
    ]
    for old_lines, new_line in param_replacements:
        status = ensure_one_of(params_lines, old_lines, new_line)
        if status == "updated":
            changes.append(f"10c cell {params_index + 1}: set {new_line}")
    params_cell["source"] = to_source(params_lines)

    if misplaced_run_markdown_index is not None and misplaced_run_lines is not None:
        previous_index = misplaced_run_markdown_index - 1
        if previous_index < 0 or notebook["cells"][previous_index].get("cell_type") != "code":
            raise RuntimeError("10c misplaced run block does not have a preceding code cell to repair")
        notebook["cells"][previous_index]["source"] = to_source(misplaced_run_lines)
        notebook["cells"][misplaced_run_markdown_index]["source"] = to_source(["## 5. Results"])
        run_index = previous_index
        changes.append(
            f"10c cell {previous_index + 1}: replaced code cell content with the authoritative main run block"
        )
        changes.append(
            f"10c cell {misplaced_run_markdown_index + 1}: restored markdown heading to ## 5. Results"
        )

    if run_index is None:
        raise RuntimeError("10c main run cell not found")

    run_cell = notebook["cells"][run_index]
    run_lines = "".join(run_cell.get("source", [])).splitlines()
    status = ensure_one_of(
        run_lines,
        [
            '    # Stage 4: Association (v75 consolidated optimal)',
            '    # Stage 4: Association (v80 best + ensemble)',
        ],
        '    # Stage 4: Association (v80 best + ensemble)',
    )
    if status == "updated":
        changes.append(f"10c cell {run_index + 1}: updated Stage 4 banner to v80 best + ensemble")

    status = ensure_one_of(
        run_lines,
        [
            '    "--override", "stage4.association.fic.regularisation=3.0",',
            '    "--override", f"stage4.association.fic.regularisation={FIC_REG}",',
            '    "--override", "stage4.association.fic.regularisation=0.1",',
        ],
        '    "--override", "stage4.association.fic.regularisation=0.1",',
    )
    if status == "updated":
        changes.append(f"10c cell {run_index + 1}: set stage4.association.fic.regularisation=0.1")

    removed_temporal = False
    removed_temporal = remove_line(run_lines, '    "--override", "stage4.association.temporal_overlap.bonus=0.10",') or removed_temporal
    removed_temporal = remove_line(run_lines, '    "--override", "stage4.association.temporal_overlap.max_mean_time=10.0",') or removed_temporal
    if removed_temporal:
        changes.append(f"10c cell {run_index + 1}: removed stale temporal_overlap bonus/max_mean_time overrides")

    new_override_lines = [
        '    "--override", f"stage4.association.gallery_expansion.enabled=true",',
        '    "--override", f"stage4.association.gallery_expansion.threshold=0.50",',
        '    "--override", f"stage4.association.weights.length_weight_power=0.3",',
        '    "--override", f"stage4.association.temporal_overlap.enabled=true",',
        '    "--override", f"stage4.association.temporal_overlap.bonus=0.05",',
        '    "--override", f"stage4.association.temporal_overlap.max_mean_time=5.0",',
    ]
    anchor = '    "--override", f"stage4.association.intra_camera_merge.max_time_gap={INTRA_MERGE_GAP}",'
    if not insert_after(run_lines, anchor, new_override_lines):
        raise RuntimeError("10c run cell missing intra_camera_merge anchor")
    if run_lines[run_lines.index(anchor) + 1:run_lines.index(anchor) + 1 + len(new_override_lines)] == new_override_lines:
        for new_line in new_override_lines:
            changes.append(f"10c cell {run_index + 1}: ensured {new_line}")

    run_cell["source"] = to_source(run_lines)

    save_notebook(NOTEBOOK_10C, notebook)
    reloaded = load_notebook(NOTEBOOK_10C)
    if not reloaded.get("cells"):
        raise RuntimeError("10c verification reload failed")
    if misplaced_run_markdown_index is not None:
        reloaded_markdown = reloaded["cells"][misplaced_run_markdown_index]
        if reloaded_markdown.get("source") != to_source(["## 5. Results"]):
            raise RuntimeError("10c results heading repair did not persist")
    return changes


def main() -> None:
    all_changes = []
    all_changes.extend(update_10a())
    all_changes.extend(update_10c())

    for path in (NOTEBOOK_10A, NOTEBOOK_10C):
        load_notebook(path)

    print("Updated notebooks:")
    for change in all_changes:
        print(f"- {change}")


if __name__ == "__main__":
    main()