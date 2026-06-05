"""Build the synthetic-augmented ResNeXt101-IBN-a training notebook (09w) from 09h.

09h trains ResNeXt101-IBN-a on real CityFlow crops only (capped ~52.77% mAP).
09w appends ~1,362 synthetic VehicleX identities to the TRAIN split (eval stays
real-only) — the AIC-winner recipe lever. We edit the proven 09h notebook in place
via json load/modify/dump (never raw text) per repo rules, then verify round-trip.
"""

import json
from pathlib import Path

SRC = Path("notebooks/kaggle/09h_resnext101ibn_dmt/09h_resnext101ibn_dmt.ipynb")
DST_DIR = Path("notebooks/kaggle/09w_resnext101ibn_synth")
DST = DST_DIR / "09w_resnext101ibn_synth.ipynb"

SYNTH_BLOCK = '''# -- Synthetic VehicleX augmentation (AIC-winner recipe lever) ----------
# AIC22 winners trained ReID on real CityFlow + ~1,362 synthetic VehicleX ids
# (85% of their training imgs were synthetic). Our real-only IBN-a backbones
# capped ~52.77% mAP. We append the synthetic identities to the TRAIN split
# only; query/gallery (eval) stay REAL-only so mAP is measured on CityFlow.
import zipfile as _zipfile

SYN_PID_OFFSET = 1_000_000
SYN_DIR = Path("/tmp/vehiclex_synth")  # ephemeral: keep 933MB OUT of kernel output
SYN_DIR.mkdir(parents=True, exist_ok=True)
import glob as _glob
# kernel_sources mount under /kaggle/input/notebooks/<owner>/<slug>/ -> search robustly
_cands = _glob.glob("/kaggle/input/**/vehiclex_vehicleid_adapted*", recursive=True)
_syn_zip = next(
    (Path(p) for p in sorted(_cands)
     if os.path.isfile(p) and os.path.getsize(p) > 1_000_000),
    None,
)
synthetic_records = []
if _syn_zip is not None:
    _marker = SYN_DIR / ".extracted"
    if not _marker.exists():
        with _zipfile.ZipFile(_syn_zip) as _zf:
            _zf.extractall(SYN_DIR)
        _marker.write_text("ok")
    for _fp in sorted(SYN_DIR.rglob("*.jpg")):
        _tok = _fp.name.split("_")[0].split(".")[0]
        if not _tok.isdigit():
            continue
        synthetic_records.append({
            "path": str(_fp),
            "pid": SYN_PID_OFFSET + int(_tok),
            "camname": "SYNTH",
            "frame_id": 0,
        })
    train_records.extend(synthetic_records)
    print(f"[synthetic] added {len(synthetic_records)} VehicleX imgs / "
          f"{len({r['pid'] for r in synthetic_records})} ids to TRAIN")
else:
    raise FileNotFoundError(
        "[synthetic] VehicleX zip not found under /kaggle/input -- this is the "
        "SYNTHETIC experiment; failing fast instead of silently training real-only."
    )

'''


def to_source(text: str) -> list[str]:
    """Jupyter source list: each line keeps its trailing newline except the last."""
    return text.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Cell 0: shorter schedule for the larger combined set + distinct output dir
    # + DISABLE center loss (it overflowed to NaN with 1828 classes in fp16).
    c0 = "".join(cells[0]["source"])
    assert '"train_epochs": 200,' in c0
    c0 = c0.replace('"train_epochs": 200,', '"train_epochs": 40,')
    assert '"center_loss_weight": 0.5,' in c0
    c0 = c0.replace('"center_loss_weight": 0.5,', '"center_loss_weight": 0.0,')
    c0 = c0.replace(
        'OUTPUT_DIR = Path("/kaggle/working/09h_output")',
        'OUTPUT_DIR = Path("/kaggle/working/09w_output")',
    )
    cells[0]["source"] = to_source(c0)

    # Cell 4: guard center loss so it is never computed/stepped when weight==0
    # (NaN*0 still propagates NaN, so we must SKIP it, not just zero-weight it).
    c4 = "".join(cells[4]["source"])
    old_loss = (
        '            loss_center = center_criterion(global_feat, labels)\n'
        '            loss = loss_id + loss_tri + CFG["center_loss_weight"] * loss_center\n'
    )
    assert old_loss in c4, "cell4 loss block anchor not found"
    c4 = c4.replace(
        old_loss,
        '            if CFG["center_loss_weight"] > 0:\n'
        '                loss_center = center_criterion(global_feat, labels)\n'
        '                loss = loss_id + loss_tri + CFG["center_loss_weight"] * loss_center\n'
        '            else:\n'
        '                loss_center = torch.zeros((), device=images.device)\n'
        '                loss = loss_id + loss_tri\n',
    )
    old_step = (
        '        for parameter in center_criterion.parameters():\n'
        '            if parameter.grad is not None:\n'
        '                parameter.grad.data *= 1.0 / max(CFG["center_loss_weight"], 1e-12)\n'
        '        center_optimizer.step()\n'
    )
    assert old_step in c4, "cell4 center-step anchor not found"
    c4 = c4.replace(
        old_step,
        '        if CFG["center_loss_weight"] > 0:\n'
        '            for parameter in center_criterion.parameters():\n'
        '                if parameter.grad is not None:\n'
        '                    parameter.grad.data *= 1.0 / max(CFG["center_loss_weight"], 1e-12)\n'
        '            center_optimizer.step()\n',
    )
    cells[4]["source"] = to_source(c4)

    # Cell 1: inject synthetic records right before the camera-id mapping.
    c1 = "".join(cells[1]["source"])
    anchor = "camname_to_id = {}\n"
    assert anchor in c1, "anchor not found in cell 1"
    c1 = c1.replace(anchor, SYNTH_BLOCK + anchor, 1)
    cells[1]["source"] = to_source(c1)

    # Cell 7: distinct output filenames + dataset label + synthetic count.
    c7 = "".join(cells[7]["source"])
    c7 = c7.replace("resnext101ibn_dmt_best.pth", "resnext101ibn_synth_best.pth")
    c7 = c7.replace("resnext101ibn_dmt_metadata.json", "resnext101ibn_synth_metadata.json")
    c7 = c7.replace("resnext101ibn_dmt_history.json", "resnext101ibn_synth_history.json")
    c7 = c7.replace(
        '"dataset": "CityFlowV2 raw GT/video crops",',
        '"dataset": "CityFlowV2 raw crops + VehicleX synthetic (VID_ReID_Simulation, ~1362 ids)",',
    )
    c7 = c7.replace(
        '"num_train_ids": len(train_pid_map),',
        '"num_train_ids": len(train_pid_map),\n    "num_synthetic_imgs": len(synthetic_records),',
    )
    cells[7]["source"] = to_source(c7)

    DST_DIR.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, ensure_ascii=True, indent=1), encoding="utf-8")

    # Verify round-trip + report.
    chk = json.loads(DST.read_text(encoding="utf-8"))
    assert len(chk["cells"]) == 8
    joined = "".join(chk["cells"][1]["source"])
    assert "synthetic_records" in joined
    c0chk = "".join(chk["cells"][0]["source"])
    assert '"train_epochs": 40,' in c0chk
    assert '"center_loss_weight": 0.0,' in c0chk
    c4chk = "".join(chk["cells"][4]["source"])
    assert 'if CFG["center_loss_weight"] > 0:' in c4chk
    print("OK wrote", DST)
    print("cells:", len(chk["cells"]))
    print("synthetic block present:", "VehicleX augmentation" in joined)


if __name__ == "__main__":
    main()
