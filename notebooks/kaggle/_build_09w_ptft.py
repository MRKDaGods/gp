"""Build the synthetic-PRETRAIN -> real-FINETUNE variant (09w-ptft) from 09w.

Idea: wrong-domain synthetic can still help if used as PRETRAINING (learn generic
vehicle/viewpoint features) then SPECIALIZED to real CityFlow. Two stages in one run:
  Stage A (epochs < stage_switch): train on combined real+synthetic
  Stage B (epochs >= stage_switch):  train on REAL-only (drop synthetic)
Same 1828-class head throughout (synthetic heads just stop getting positives in B).
Eval stays real-only. This is the single best shot at making the accessible
(VehicleID-adapted) synthetic actually beat the real-only baseline.
"""

import json
from pathlib import Path

SRC = Path("notebooks/kaggle/09w_resnext101ibn_synth/09w_resnext101ibn_synth.ipynb")
DST_DIR = Path("notebooks/kaggle/09w_ptft")
DST = DST_DIR / "09w_ptft.ipynb"


def to_source(text):
    return text.splitlines(keepends=True)


REAL_LOADER_CELL = '''# Stage-B real-only loader (finetune stage drops synthetic identities).
real_train_records = [r for r in train_records if r.get("camname") != "SYNTH"]
train_loader_real = DataLoader(
    ReIDImageDataset(real_train_records, train_transform),
    batch_size=TRAIN_BATCH_SIZE,
    sampler=RandomIdentitySampler(real_train_records, CFG["batch_p"], CFG["batch_k"]),
    num_workers=CFG["num_workers"],
    pin_memory=True,
    drop_last=True,
)
print(
    f"[ptft] Stage-B real-only loader: {len(real_train_records)} imgs / "
    f"{len({r['pid'] for r in real_train_records})} ids; switch @ epoch "
    f"{CFG['stage_switch_epoch']}"
)
'''


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Cell 0: add stage_switch_epoch + distinct output dir.
    c0 = "".join(cells[0]["source"])
    assert '"train_epochs": 40,' in c0
    c0 = c0.replace(
        '"train_epochs": 40,',
        '"train_epochs": 40,\n    "stage_switch_epoch": 25,',
    )
    c0 = c0.replace("09w_output", "09wptft_output")
    cells[0]["source"] = to_source(c0)

    # New cell after cell 1: build the real-only loader.
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(REAL_LOADER_CELL),
    }
    cells.insert(2, new_cell)

    # Cell 4 (now index 5 after insert): stage-switch the loader in the epoch loop.
    c5 = "".join(cells[5]["source"])
    anchor = (
        "    train_metrics = train_one_epoch(\n"
        "        model,\n"
        "        train_loader,\n"
    )
    assert anchor in c5, "epoch-loop loader anchor not found"
    c5 = c5.replace(
        anchor,
        '    cur_loader = (\n'
        '        train_loader if epoch < CFG["stage_switch_epoch"] else train_loader_real\n'
        '    )\n'
        '    if epoch == CFG["stage_switch_epoch"]:\n'
        '        print(f"[ptft] === Stage B: real-only finetune from epoch {epoch} ===")\n'
        "    train_metrics = train_one_epoch(\n"
        "        model,\n"
        "        cur_loader,\n",
    )
    cells[5]["source"] = to_source(c5)

    # Cell 7 (now index 8): retag output filenames.
    c8 = "".join(cells[8]["source"])
    c8 = c8.replace("resnext101ibn_synth_", "resnext101ibn_ptft_")
    cells[8]["source"] = to_source(c8)

    DST_DIR.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, ensure_ascii=True, indent=1), encoding="utf-8")

    meta = {
        "id": "yahiaakhalafallah/09w-ptft",
        "title": "09w ptft",
        "code_file": "09w_ptft.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": ["thanhnguyenle/data-aicity-2023-track-2"],
        "kernel_sources": ["yahiaakhalafallah/vehiclex-probe"],
        "competition_sources": [],
    }
    (DST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    # Verify round-trip + syntax.
    import ast

    chk = json.loads(DST.read_text(encoding="utf-8"))
    bad = False
    for i, c in enumerate(chk["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        ls = ["" if l.lstrip().startswith("!") else l for l in s.split("\n")]
        try:
            ast.parse("\n".join(ls))
        except SyntaxError as e:
            bad = True
            print("SYN ERR cell", i, e)
    print("cells:", len(chk["cells"]), "syntax:", "BAD" if bad else "OK")
    print("stage_switch present:", '"stage_switch_epoch": 25,' in "".join(chk["cells"][0]["source"]))
    print("real loader cell present:", "train_loader_real" in "".join(chk["cells"][2]["source"]))
    print("staged loop present:", "cur_loader" in "".join(chk["cells"][5]["source"]))


if __name__ == "__main__":
    main()
