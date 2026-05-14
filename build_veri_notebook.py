"""Build the 09v VeRi-776 evaluation Kaggle notebook (v2: dynamic path discovery)."""
import json
from pathlib import Path

OUT = Path("notebooks/kaggle/09v_veri776_eval/09v-veri776-eval.ipynb")

def code_cell(src):
    lines = src.split("\n")
    source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else [""]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

def md_cell(src):
    lines = src.split("\n")
    source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else [""]
    return {"cell_type": "markdown", "metadata": {}, "source": source}

cells = []

cells.append(md_cell(
    "# 09v VeRi-776 Evaluation: TransReID ViT-B/16 (CLIP) + Reranking\n\n"
    "Goal: produce a verifiable mAP / R1 number for our deployed TransReID checkpoint on VeRi-776.\n\n"
    "Eval params: `qe_k=3`, `k1=30`, `k2=10`, `lambda_value=0.2`."
))

cells.append(code_cell(
    "import sys, os, subprocess\n"
    "print(\"Python:\", sys.version)\n"
    "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"timm==1.0.11\", \"loguru\"], check=True)\n"
    "print(\"Deps installed\")"
))

cells.append(code_cell(
    "# Discover mounted input directories (paths can vary)\n"
    "import os\n"
    "for root in sorted(os.listdir(\"/kaggle/input\")):\n"
    "    full = os.path.join(\"/kaggle/input\", root)\n"
    "    print(f\"{full}/\")\n"
    "    try:\n"
    "        for sub in sorted(os.listdir(full))[:8]:\n"
    "            print(f\"  -> {sub}\")\n"
    "    except Exception as e:\n"
    "        print(f\"  ERR: {e}\")"
))

cells.append(code_cell(
    "# Resolve VeRi root and weights dir robustly\n"
    "from pathlib import Path\n"
    "import os\n\n"
    "def find_veri_root():\n"
    "    # Try common patterns\n"
    "    candidates = [\n"
    "        \"/kaggle/input/veri-vehicle-re-identification-dataset/VeRi\",\n"
    "        \"/kaggle/input/veri-vehicle-re-identification-dataset\",\n"
    "    ]\n"
    "    for base in os.listdir(\"/kaggle/input\"):\n"
    "        candidates.append(f\"/kaggle/input/{base}/VeRi\")\n"
    "        candidates.append(f\"/kaggle/input/{base}\")\n"
    "    for c in candidates:\n"
    "        p = Path(c)\n"
    "        if (p / \"image_query\").is_dir() and (p / \"image_test\").is_dir():\n"
    "            return p\n"
    "    raise RuntimeError(\"Could not locate VeRi-776 dataset (need image_query and image_test subdirs)\")\n\n"
    "def find_weights(filename):\n"
    "    # Search recursively under /kaggle/input for the checkpoint\n"
    "    for base in os.listdir(\"/kaggle/input\"):\n"
    "        for root, _, files in os.walk(f\"/kaggle/input/{base}\"):\n"
    "            if filename in files:\n"
    "                return os.path.join(root, filename)\n"
    "    raise RuntimeError(f\"Could not locate {filename}\")\n\n"
    "VERI_ROOT = find_veri_root()\n"
    "WEIGHTS_VERI = find_weights(\"vehicle_transreid_vit_base_veri776.pth\")\n"
    "WEIGHTS_CITY = find_weights(\"transreid_cityflowv2_best.pth\")\n"
    "print(\"VeRi root  :\", VERI_ROOT)\n"
    "print(\"VeRi ckpt  :\", WEIGHTS_VERI)\n"
    "print(\"CityFlow   :\", WEIGHTS_CITY)"
))

cells.append(code_cell(
    "# Clone the gp repo to access src/training/evaluate_reid.py and src/stage2_features/transreid_model.py\n"
    "import subprocess, os, sys\n"
    "if not os.path.isdir(\"/kaggle/working/gp\"):\n"
    "    subprocess.run([\"git\", \"clone\", \"--depth\", \"1\", \"https://github.com/MRKDaGods/gp.git\", \"/kaggle/working/gp\"], check=True)\n"
    "if \"/kaggle/working/gp\" not in sys.path:\n"
    "    sys.path.insert(0, \"/kaggle/working/gp\")\n"
    "print(os.listdir(\"/kaggle/working/gp/src\")[:10])"
))

cells.append(code_cell(
    "# Parse VeRi-776 query / gallery\n"
    "from pathlib import Path\n\n"
    "def parse_split(split_dir):\n"
    "    data = []\n"
    "    pid_set = set()\n"
    "    for img_path in sorted(Path(split_dir).glob(\"*.jpg\")):\n"
    "        parts = img_path.stem.split(\"_\")\n"
    "        if len(parts) < 2:\n"
    "            continue\n"
    "        pid = int(parts[0])\n"
    "        if pid == -1:\n"
    "            continue  # junk\n"
    "        camid = int(parts[1][1:]) - 1\n"
    "        pid_set.add(pid)\n"
    "        data.append((str(img_path), pid, camid))\n"
    "    return data, len(pid_set)\n\n"
    "query, n_q = parse_split(VERI_ROOT / \"image_query\")\n"
    "gallery, n_g = parse_split(VERI_ROOT / \"image_test\")\n"
    "print(f\"Query:   {len(query):,} images, {n_q} IDs\")\n"
    "print(f\"Gallery: {len(gallery):,} images, {n_g} IDs\")"
))

cells.append(code_cell(
    "# Dataloader\n"
    "import torch\n"
    "import torchvision.transforms as T\n"
    "from torch.utils.data import Dataset, DataLoader\n"
    "from PIL import Image\n\n"
    "CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]\n"
    "CLIP_STD = [0.26862954, 0.26130258, 0.27577711]\n\n"
    "class VeRiDataset(Dataset):\n"
    "    def __init__(self, items, transform):\n"
    "        self.items = items\n"
    "        self.transform = transform\n"
    "    def __len__(self):\n"
    "        return len(self.items)\n"
    "    def __getitem__(self, i):\n"
    "        path, pid, camid = self.items[i]\n"
    "        img = Image.open(path).convert(\"RGB\")\n"
    "        return self.transform(img), pid, camid, path\n\n"
    "IMG_SIZE = (256, 256)\n"
    "transform = T.Compose([\n"
    "    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),\n"
    "    T.ToTensor(),\n"
    "    T.Normalize(CLIP_MEAN, CLIP_STD),\n"
    "])\n"
    "q_loader = DataLoader(VeRiDataset(query, transform), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)\n"
    "g_loader = DataLoader(VeRiDataset(gallery, transform), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)\n"
    "print(\"Loaders ready\")"
))

cells.append(code_cell(
    "from src.stage2_features.transreid_model import build_transreid\n"
    "from src.training.evaluate_reid import (\n"
    "    extract_features, compute_distance_matrix, eval_market1501, compute_reranking,\n"
    ")\n"
    "import numpy as np\n"
    "DEVICE = \"cuda:0\" if torch.cuda.is_available() else \"cpu\"\n"
    "print(\"Device:\", DEVICE)\n\n"
    "def load_model(weights_path):\n"
    "    m = build_transreid(\n"
    "        num_classes=1, num_cameras=20, embed_dim=768,\n"
    "        vit_model=\"vit_base_patch16_clip_224.openai\",\n"
    "        pretrained=False, weights_path=weights_path, img_size=IMG_SIZE,\n"
    "    )\n"
    "    return m.to(DEVICE).eval()"
))

cells.append(code_cell(
    "import numpy as np\n"
    "def average_query_expansion(features, k):\n"
    "    if k <= 0:\n"
    "        return features\n"
    "    sim = features @ features.T\n"
    "    topk_idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]\n"
    "    expanded = np.zeros_like(features)\n"
    "    for i in range(len(features)):\n"
    "        expanded[i] = features[topk_idx[i]].mean(axis=0)\n"
    "    norms = np.linalg.norm(expanded, axis=1, keepdims=True) + 1e-12\n"
    "    return expanded / norms"
))

cells.append(code_cell(
    "import time\n"
    "def run_full_eval(weights_path, label):\n"
    "    print(f\"\\n{'='*70}\\n{label}\\n  weights={weights_path}\\n{'='*70}\")\n"
    "    model = load_model(weights_path)\n"
    "    t0 = time.time()\n"
    "    q_f, q_p, q_c = extract_features(model, q_loader, DEVICE)\n"
    "    g_f, g_p, g_c = extract_features(model, g_loader, DEVICE)\n"
    "    print(f\"  Features extracted in {time.time()-t0:.1f}s -- q={q_f.shape}, g={g_f.shape}\")\n"
    "    results = {}\n"
    "    distmat = compute_distance_matrix(q_f, g_f, metric=\"cosine\")\n"
    "    mAP, cmc = eval_market1501(distmat, q_p, g_p, q_c, g_c)\n"
    "    results[\"baseline\"] = (mAP, cmc[0], cmc[4], cmc[9])\n"
    "    print(f\"  [Baseline]   mAP={mAP*100:.4f}%  R1={cmc[0]*100:.4f}%  R5={cmc[4]*100:.2f}%  R10={cmc[9]*100:.2f}%\")\n"
    "    qe_k = 3\n"
    "    all_f = np.concatenate([q_f, g_f], axis=0)\n"
    "    all_f_qe = average_query_expansion(all_f, k=qe_k)\n"
    "    q_f_qe, g_f_qe = all_f_qe[:len(q_p)], all_f_qe[len(q_p):]\n"
    "    distmat_qe = compute_distance_matrix(q_f_qe, g_f_qe, metric=\"cosine\")\n"
    "    mAP_qe, cmc_qe = eval_market1501(distmat_qe, q_p, g_p, q_c, g_c)\n"
    "    results[\"aqe\"] = (mAP_qe, cmc_qe[0], cmc_qe[4], cmc_qe[9])\n"
    "    print(f\"  [AQE k=3]    mAP={mAP_qe*100:.4f}%  R1={cmc_qe[0]*100:.4f}%  R5={cmc_qe[4]*100:.2f}%  R10={cmc_qe[9]*100:.2f}%\")\n"
    "    print(\"  computing rerank (k1=30,k2=10,lambda=0.2) ...\")\n"
    "    t1 = time.time()\n"
    "    distmat_rr = compute_reranking(q_f, g_f, k1=30, k2=10, lambda_value=0.2)\n"
    "    mAP_rr, cmc_rr = eval_market1501(distmat_rr, q_p, g_p, q_c, g_c)\n"
    "    results[\"rerank\"] = (mAP_rr, cmc_rr[0], cmc_rr[4], cmc_rr[9])\n"
    "    print(f\"  [Rerank]     mAP={mAP_rr*100:.4f}%  R1={cmc_rr[0]*100:.4f}%  R5={cmc_rr[4]*100:.2f}%  R10={cmc_rr[9]*100:.2f}%  ({time.time()-t1:.1f}s)\")\n"
    "    print(\"  computing AQE+rerank ...\")\n"
    "    t2 = time.time()\n"
    "    distmat_qrr = compute_reranking(q_f_qe, g_f_qe, k1=30, k2=10, lambda_value=0.2)\n"
    "    mAP_qrr, cmc_qrr = eval_market1501(distmat_qrr, q_p, g_p, q_c, g_c)\n"
    "    results[\"aqe_rerank\"] = (mAP_qrr, cmc_qrr[0], cmc_qrr[4], cmc_qrr[9])\n"
    "    print(f\"  [AQE+Rerank] mAP={mAP_qrr*100:.4f}%  R1={cmc_qrr[0]*100:.4f}%  R5={cmc_qrr[4]*100:.2f}%  R10={cmc_qrr[9]*100:.2f}%  ({time.time()-t2:.1f}s)\")\n"
    "    del model\n"
    "    torch.cuda.empty_cache()\n"
    "    return results"
))

cells.append(code_cell(
    "# Primary eval: VeRi-776-trained checkpoint\n"
    "results_veri = run_full_eval(WEIGHTS_VERI, \"Checkpoint A: vehicle_transreid_vit_base_veri776.pth (VeRi-776 trained)\")"
))

cells.append(code_cell(
    "# Sanity check: CityFlowV2-fine-tuned checkpoint\n"
    "results_city = run_full_eval(WEIGHTS_CITY, \"Checkpoint B: transreid_cityflowv2_best.pth (CityFlowV2 fine-tuned)\")"
))

cells.append(code_cell(
    "import json\n"
    "summary = {\n"
    "    \"img_size\": list(IMG_SIZE),\n"
    "    \"vit_model\": \"vit_base_patch16_clip_224.openai\",\n"
    "    \"flip_aug\": True,\n"
    "    \"rerank_params\": {\"qe_k\": 3, \"k1\": 30, \"k2\": 10, \"lambda_value\": 0.2},\n"
    "    \"checkpoints\": {\n"
    "        \"vehicle_transreid_vit_base_veri776.pth\": {cfg: {\"mAP\": float(v[0]), \"R1\": float(v[1]), \"R5\": float(v[2]), \"R10\": float(v[3])} for cfg, v in results_veri.items()},\n"
    "        \"transreid_cityflowv2_best.pth\":          {cfg: {\"mAP\": float(v[0]), \"R1\": float(v[1]), \"R5\": float(v[2]), \"R10\": float(v[3])} for cfg, v in results_city.items()},\n"
    "    },\n"
    "}\n"
    "with open(\"/kaggle/working/veri776_eval_results.json\", \"w\") as f:\n"
    "    json.dump(summary, f, indent=2)\n"
    "print(json.dumps(summary, indent=2))"
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"Wrote {OUT}: {OUT.stat().st_size} bytes, {len(cells)} cells")
