"""Update 10c for v49 push: enable scan + replace CSLS A/B with temporal_split/cluster_verify."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
nb_path = ROOT / "notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

changes = 0

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])

    # 1. Enable scan
    if "SCAN_ENABLED = False" in src:
        nb["cells"][i]["source"] = [
            line.replace("SCAN_ENABLED = False", "SCAN_ENABLED = True")
            for line in cell["source"]
        ]
        print(f"[Cell {i}] SCAN_ENABLED = True")
        changes += 1

    # 2. Replace CSLS A/B tests with temporal_split + cluster_verify
    if "FEATURE_TEST_ENABLED = True" in src and "csls_k10" in src:
        new_experiments = '''    feature_experiments = [
        # --- temporal_split: split clusters at time gaps (targets conflation) ---
        ("tsplit_gap60_t050", {"stage4.association.temporal_split.enabled": "true",
                               "stage4.association.temporal_split.min_gap": "60",
                               "stage4.association.temporal_split.split_threshold": "0.50"}),
        ("tsplit_gap45_t045", {"stage4.association.temporal_split.enabled": "true",
                               "stage4.association.temporal_split.min_gap": "45",
                               "stage4.association.temporal_split.split_threshold": "0.45"}),
        ("tsplit_gap30_t040", {"stage4.association.temporal_split.enabled": "true",
                               "stage4.association.temporal_split.min_gap": "30",
                               "stage4.association.temporal_split.split_threshold": "0.40"}),
        # --- cluster_verify: eject weakly-connected cluster members ---
        ("cverify_030", {"stage4.association.cluster_verify.enabled": "true",
                         "stage4.association.cluster_verify.min_connectivity": "0.30"}),
        ("cverify_035", {"stage4.association.cluster_verify.enabled": "true",
                         "stage4.association.cluster_verify.min_connectivity": "0.35"}),
        ("cverify_025", {"stage4.association.cluster_verify.enabled": "true",
                         "stage4.association.cluster_verify.min_connectivity": "0.25"}),
        # --- combined: temporal_split + cluster_verify ---
        ("tsplit60_cverify030", {"stage4.association.temporal_split.enabled": "true",
                                 "stage4.association.temporal_split.min_gap": "60",
                                 "stage4.association.temporal_split.split_threshold": "0.50",
                                 "stage4.association.cluster_verify.enabled": "true",
                                 "stage4.association.cluster_verify.min_connectivity": "0.30"}),
    ]'''
        old_experiments_start = '    feature_experiments = ['
        old_experiments_end = '    ]\n    feat_results = []'
        
        lines = cell["source"]
        new_lines = []
        skip = False
        replaced = False
        for line in lines:
            if old_experiments_start in line and not replaced:
                skip = True
                # Insert new experiments block
                for new_line in new_experiments.split("\n"):
                    new_lines.append(new_line + "\n")
                replaced = True
                continue
            if skip:
                if "feat_results = []" in line:
                    skip = False
                    new_lines.append("    feat_results = []\n")
                continue
            new_lines.append(line)
        
        if replaced:
            nb["cells"][i]["source"] = new_lines
            print(f"[Cell {i}] Replaced CSLS experiments with temporal_split + cluster_verify (7 tests)")
            changes += 1

if changes > 0:
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"\nSaved {changes} changes to 10c notebook")
else:
    print("No changes needed!")

print("Done. Review with: git diff --stat")
