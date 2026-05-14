from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from textwrap import dedent, indent


NOTEBOOK_PATH = Path(
    r"e:\dev\src\gp\notebooks\kaggle\09d_vehicle_reid_resnet101ibn\09d_vehicle_reid_resnet101ibn.ipynb"
)
TRAIN_SCRIPT_PATH = "/kaggle/working/train_09d.py"
VERI776_CHECKPOINT = "/kaggle/input/mtmc-weights/reid/resnet101ibn_veri776_best.pth"


def to_source(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    return [f"{line}\n" for line in lines[:-1]] + [lines[-1]]


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def strip_leading_imports(text: str) -> str:
    cleaned_lines: list[str] = []
    in_leading_block = True
    for line in text.splitlines():
        stripped = line.strip()
        if in_leading_block and (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        if in_leading_block and not stripped:
            continue
        if stripped and not stripped.startswith("#"):
            in_leading_block = False
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def sanitize_training_script(text: str) -> str:
    lines = text.splitlines()
    try:
        main_start = next(index for index, line in enumerate(lines) if line.startswith("def main():"))
    except StopIteration:
        return text if text.endswith("\n") else text + "\n"

    main_end = len(lines)
    for index in range(main_start + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t")):
            main_end = index
            break

    module_lines = lines[:main_start]
    main_lines = lines[main_start:main_end]
    tail_lines = lines[main_end:]

    local_imports: list[str] = []
    sanitized_main_lines = [main_lines[0]]
    for line in main_lines[1:]:
        stripped = line.lstrip()
        indent_width = len(line) - len(stripped)
        if indent_width == 4 and (stripped.startswith("import ") or stripped.startswith("from ")):
            local_imports.append(stripped)
            continue
        sanitized_main_lines.append(line)

    existing_module_imports = {
        line.strip() for line in module_lines if line.startswith("import ") or line.startswith("from ")
    }
    missing_imports = [entry for entry in local_imports if entry not in existing_module_imports]

    insert_at = 0
    if module_lines and module_lines[0].startswith("#!"):
        insert_at = 1
    if insert_at < len(module_lines) and module_lines[insert_at].startswith(('"""', "'''")):
        quote = module_lines[insert_at][:3]
        while insert_at < len(module_lines):
            line = module_lines[insert_at]
            insert_at += 1
            if line.endswith(quote) and len(line) > 3:
                break
            if line.strip().endswith(quote) and line.strip() != quote:
                break
            if line.strip() == quote:
                break
    while insert_at < len(module_lines) and not module_lines[insert_at].strip():
        insert_at += 1
    while insert_at < len(module_lines) and (
        module_lines[insert_at].startswith("import ") or module_lines[insert_at].startswith("from ")
    ):
        insert_at += 1

    if missing_imports:
        prefix = module_lines[:insert_at]
        suffix = module_lines[insert_at:]
        if prefix and prefix[-1].strip():
            prefix.append("")
        prefix.extend(missing_imports)
        if suffix and suffix[0].strip():
            prefix.append("")
        module_lines = prefix + suffix

    sanitized_lines = module_lines + sanitized_main_lines + tail_lines
    sanitized_text = "\n".join(sanitized_lines).rstrip() + "\n"

    replacements = {
        "def eval_reid(qf, qp, qc, gf, gp, gc, max_rank=50):": "def eval_reid(qf, qp, query_cams, gf, gp, gallery_cams, max_rank=50):",
        "valid = ~((gp[order] == qp[i]) & (gc[order] == qc[i]))": "valid = ~((gp[order] == qp[i]) & (gallery_cams[order] == query_cams[i]))",
        "gf, gp, gc = extract_features(model, gallery_loader, DEVICE)": "gf, gp, gallery_cams = extract_features(model, gallery_loader, DEVICE)",
        "mAP, cmc = eval_reid(qf, qp, qc, gf, gp, gc)": "mAP, cmc = eval_reid(qf, qp, qc, gf, gp, gallery_cams)",
    }
    for old, new in replacements.items():
        sanitized_text = sanitized_text.replace(old, new)

    backbone_marker = '"backbone_lr_factor": 0.1,'
    if "backbone_lr_factor" in sanitized_text and backbone_marker not in sanitized_text:
        lr_line = '    "lr": 3.5e-4,'
        if lr_line in sanitized_text:
            sanitized_text = sanitized_text.replace(
                lr_line,
                lr_line + "\n" + '    "backbone_lr_factor": 0.1,',
                1,
            )

    return sanitized_text


def compile_python(source: str, filename: str) -> None:
    compile(source, filename, "exec")


def find_main_local_bindings(source: str) -> list[str]:
    tracked_names = {
        "os",
        "sys",
        "time",
        "json",
        "cv2",
        "np",
        "numpy",
        "torch",
        "shutil",
        "gc",
        "socket",
        "traceback",
    }

    tree = ast.parse(source)
    main_node = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
        None,
    )
    if main_node is None:
        return []

    findings: set[str] = set()

    def record_name(name: str, lineno: int, kind: str) -> None:
        if name in tracked_names:
            findings.add(f"line {lineno}: {kind} {name}")

    for node in ast.walk(main_node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                record_name(alias.asname or alias.name.split(".", 1)[0], node.lineno, "import")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                record_name(alias.asname or alias.name, node.lineno, "import")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for child in ast.walk(target):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        record_name(child.id, child.lineno, "assignment")
        elif isinstance(node, ast.AnnAssign):
            for child in ast.walk(node.target):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    record_name(child.id, child.lineno, "assignment")
        elif isinstance(node, ast.AugAssign):
            for child in ast.walk(node.target):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    record_name(child.id, child.lineno, "assignment")

    return sorted(findings)


def clean_import_cell(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "%%capture":
            continue
        if stripped.startswith("!pip install"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def build_cell1() -> str:
    return dedent(
        """
        import os
        import sys
        import subprocess
        import shutil

        def run_pip(args):
            command = [sys.executable, "-m", "pip", "install"] + args
            print("[bootstrap] Running:", " ".join(command))
            subprocess.check_call(command)

        def detect_gpu_compute_caps():
            nvidia_smi = shutil.which("nvidia-smi")
            if not nvidia_smi:
                print("[bootstrap] nvidia-smi not found; skipping GPU capability detection")
                return []

            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=gpu_name,compute_cap",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("[bootstrap] GPU detection failed:")
                print(result.stderr.strip() or result.stdout.strip())
                return []

            entries = []
            for raw_line in result.stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                name, _, cap_text = line.partition(",")
                name = name.strip()
                cap_text = cap_text.strip()
                try:
                    capability = float(cap_text)
                except ValueError:
                    print(f"[bootstrap] Could not parse compute capability from: {line}")
                    continue
                entries.append((name, capability))
            return entries

        gpu_entries = detect_gpu_compute_caps()
        if gpu_entries:
            for gpu_name, capability in gpu_entries:
                print(f"[bootstrap] Detected GPU: {gpu_name} (compute capability {capability:.1f})")
        else:
            print("[bootstrap] No GPU capability entries detected")

        needs_p100_compatible_torch = any(capability < 7.0 for _, capability in gpu_entries)
        if needs_p100_compatible_torch:
            print("[bootstrap] Detected pre-Volta GPU; installing PyTorch 2.4.1/cu124 for P100 compatibility")
            run_pip([
                "torch==2.4.1",
                "torchvision==0.19.1",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
            ])
        else:
            print("[bootstrap] Torch downgrade not required")

        print("[bootstrap] Installing notebook dependencies")
        run_pip(["timm==0.9.16", "loguru", "omegaconf", "torchreid", "opencv-python-headless"])
        print("[bootstrap] Environment setup complete")
        """
    ).strip()


def build_model_block() -> str:
    return dedent(
        f"""
        IBN_NET_RESNET101_A_URL = "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth"
        IBN_NET_RESNET101_A_LOCAL_PATH = "/kaggle/working/resnet101_ibn_a.pth"
        STEP6_MODEL_ERROR_PATH = "/kaggle/working/STEP6_MODEL_ERROR.txt"


        class IBN_a(nn.Module):
            def __init__(self, planes):
                super().__init__()
                half = planes // 2
                self.IN = nn.InstanceNorm2d(half, affine=True)
                self.BN = nn.BatchNorm2d(planes - half)

            def forward(self, x):
                split = x.shape[1] // 2
                return torch.cat([self.IN(x[:, :split]), self.BN(x[:, split:])], dim=1)


        class GeM(nn.Module):
            def __init__(self, p=3.0, eps=1e-6):
                super().__init__()
                self.p = nn.Parameter(torch.ones(1) * p)
                self.eps = eps

            def forward(self, x):
                return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


        class ResNet101IBN(nn.Module):
            def __init__(self, num_classes, feat_dim=2048, last_stride=1, gem_p=3.0, pretrained=True):
                super().__init__()
                base = tv_models.resnet101(weights=None)
                for layer in [base.layer1, base.layer2, base.layer3]:
                    for block in layer:
                        if hasattr(block, "bn1"):
                            block.bn1 = IBN_a(block.bn1.num_features)
                load_result = None
                if pretrained:
                    state_dict = None
                    if os.path.exists(IBN_NET_RESNET101_A_LOCAL_PATH):
                        print(f"Using cached pretrained weights: {{IBN_NET_RESNET101_A_LOCAL_PATH}}")
                    else:
                        try:
                            print(f"Downloading pretrained weights from {{IBN_NET_RESNET101_A_URL}}")
                            print(f"Saving pretrained weights to {{IBN_NET_RESNET101_A_LOCAL_PATH}}")
                            original_timeout = socket.getdefaulttimeout()
                            socket.setdefaulttimeout(120)
                            try:
                                urllib.request.urlretrieve(
                                    IBN_NET_RESNET101_A_URL,
                                    IBN_NET_RESNET101_A_LOCAL_PATH,
                                )
                            finally:
                                socket.setdefaulttimeout(original_timeout)
                            print(f"Pretrained weights downloaded to {{IBN_NET_RESNET101_A_LOCAL_PATH}}")
                        except Exception as download_error:
                            print(f"Direct download failed: {{download_error}}")
                            print("Falling back to torch.hub.load_state_dict_from_url")
                            state_dict = torch.hub.load_state_dict_from_url(
                                IBN_NET_RESNET101_A_URL,
                                map_location="cpu",
                                progress=True,
                            )
                    if state_dict is None:
                        print(f"Loading pretrained weights from {{IBN_NET_RESNET101_A_LOCAL_PATH}}")
                        state_dict = torch.load(IBN_NET_RESNET101_A_LOCAL_PATH, map_location="cpu")
                    load_result = base.load_state_dict(state_dict, strict=False)
                    print(f"Missing keys: {{load_result.missing_keys}}")
                    print(f"Unexpected keys: {{load_result.unexpected_keys}}")
                if load_result is not None:
                    assert set(load_result.missing_keys) <= {{"fc.weight", "fc.bias"}}, f"Unexpected missing keys: {{load_result.missing_keys}}"
                    allowed_unexpected_keys = {{"fc.weight", "fc.bias", "classifier.weight", "classifier.bias"}}
                    assert set(load_result.unexpected_keys) <= allowed_unexpected_keys, f"Unexpected extra keys: {{load_result.unexpected_keys}}"
                if last_stride == 1:
                    for module in base.layer4.modules():
                        if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                            module.stride = (1, 1)
                self.conv1 = base.conv1
                self.bn1 = base.bn1
                self.relu = base.relu
                self.maxpool = base.maxpool
                self.layer1 = base.layer1
                self.layer2 = base.layer2
                self.layer3 = base.layer3
                self.layer4 = base.layer4
                self.pool = GeM(p=gem_p)
                self.feat_dim = feat_dim
                self.bottleneck = nn.BatchNorm1d(feat_dim)
                self.bottleneck.bias.requires_grad_(False)
                nn.init.constant_(self.bottleneck.weight, 1)
                nn.init.constant_(self.bottleneck.bias, 0)
                self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
                nn.init.normal_(self.classifier.weight, std=0.001)

            def forward(self, x):
                x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                global_feat = self.pool(x).view(x.size(0), -1)
                bn_feat = self.bottleneck(global_feat)
                if self.training:
                    return self.classifier(bn_feat), global_feat, bn_feat
                return F.normalize(bn_feat, p=2, dim=1)


        try:
            model_kwargs = dict(
                num_classes=NUM_CLASSES,
                feat_dim=CFG["feat_dim"],
                last_stride=1,
                gem_p=CFG["gem_p"],
            )
            if os.path.exists(VERI776_CHECKPOINT):
                print(f"Loading VeRi-776 pretrained weights from {{VERI776_CHECKPOINT}}")
                model = ResNet101IBN(pretrained=False, **model_kwargs)
                veri_state = torch.load(VERI776_CHECKPOINT, map_location="cpu", weights_only=False)
                if isinstance(veri_state, dict) and "state_dict" in veri_state:
                    veri_state = veri_state["state_dict"]
                veri_state = {{
                    key[len("module."):] if key.startswith("module.") else key: value
                    for key, value in veri_state.items()
                }}
                model_state = model.state_dict()
                loaded, skipped = [], []
                for key, value in veri_state.items():
                    if key in model_state and hasattr(value, "shape") and value.shape == model_state[key].shape:
                        model_state[key] = value
                        loaded.append(key)
                    else:
                        skipped.append(key)
                model.load_state_dict(model_state)
                print(f"Loaded {{len(loaded)}} VeRi-776 keys; skipped {{len(skipped)}}")
                if skipped:
                    print(f"Skipped keys sample: {{skipped[:10]}}")
            else:
                print(f"VeRi-776 checkpoint not found at {{VERI776_CHECKPOINT}}; using ImageNet initialization")
                model = ResNet101IBN(pretrained=True, **model_kwargs)
            model = model.to(DEVICE)
            if torch.cuda.device_count() > 1:
                print(f"Using DataParallel on {{torch.cuda.device_count()}} GPUs")
                model = nn.DataParallel(model)

            def unwrap_model(model):
                return model.module if hasattr(model, "module") else model

            def get_model_state_dict(model):
                return unwrap_model(model).state_dict()

            def load_model_state_dict(model, state_dict):
                if any(key.startswith("module.") for key in state_dict):
                    state_dict = {{key[len("module."):]: value for key, value in state_dict.items()}}
                unwrap_model(model).load_state_dict(state_dict)

            num_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"ResNet101-IBN-a params: {{num_params:.1f}}M")
        except Exception:
            error_text = traceback.format_exc()
            with open(STEP6_MODEL_ERROR_PATH, "w", encoding="utf-8") as handle:
                handle.write(error_text)
            print(f"STEP6 model creation failed. Full traceback written to {{STEP6_MODEL_ERROR_PATH}}")
            raise
        """
    ).strip()


def extract_training_script_from_rebuilt_notebook(cells: list[dict]) -> str | None:
    if len(cells) != 3:
        return None
    cell2_text = cell_text(cells[1])
    match = re.search(r"script_content\s*=\s*r'''(.*)'''\s*script_path\s*=", cell2_text, flags=re.S)
    if match is None:
        return None
    return match.group(1)


def build_training_script(original_cells: list[dict]) -> str:
    rebuilt_script = extract_training_script_from_rebuilt_notebook(original_cells)
    if rebuilt_script is not None:
        return sanitize_training_script(rebuilt_script)

    if len(original_cells) < 16:
        raise RuntimeError(f"Expected at least 16 cells, found {len(original_cells)}")

    if 'CFG = {' not in cell_text(original_cells[2]):
        raise RuntimeError("Unexpected notebook layout: CFG cell not found at index 2")
    if 'class ResNet101IBN(nn.Module):' not in cell_text(original_cells[8]):
        raise RuntimeError("Unexpected notebook layout: model cell not found at index 8")

    imports_block = dedent(
        '''
        #!/usr/bin/env python3
        """09d Vehicle ReID ResNet101-IBN-a Training Script (subprocess-safe for P100)"""

        import os
        import sys
        import time
        import json
        import cv2
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from PIL import Image
        import torchvision.transforms as T
        import torchvision.models as tv_models
        from loguru import logger
        from pathlib import Path
        from collections import defaultdict
        from torch.utils.data import Dataset, DataLoader, Sampler
        import shutil
        import gc
        import socket
        import urllib.request
        import traceback
        import atexit
        '''
    ).strip()

    debug_block = dedent(
        """
        _LOG_PATH = Path("/kaggle/working/debug.log")
        _LOG = open(_LOG_PATH, "w", buffering=1)

        class TeeWriter:
            def __init__(self, orig, log):
                self.orig = orig
                self.log = log

            def write(self, s):
                self.orig.write(s)
                try:
                    self.log.write(s)
                except Exception:
                    pass

            def flush(self):
                self.orig.flush()
                try:
                    self.log.flush()
                except Exception:
                    pass
        """
    ).strip()

    main_sections = [
        dedent(
            f"""
            sys.stdout = TeeWriter(sys.stdout, _LOG)
            sys.stderr = TeeWriter(sys.stderr, _LOG)

            def _shutdown():
                try:
                    _LOG.close()
                except Exception:
                    pass

            atexit.register(_shutdown)

            print("[DEBUG] Logging initialized")
            print(f"[DEBUG] Hostname: {{os.uname().nodename if hasattr(os, 'uname') else 'unknown'}}")

            DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Device: {{DEVICE}}, GPU: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}}")
            print(f"PyTorch: {{torch.__version__}}")

            VERI776_CHECKPOINT = "{VERI776_CHECKPOINT}"

            CFG = {{
                "dataset_root": "/kaggle/working/cityflowv2_reid",
                "weights_output": "/kaggle/working/resnet101ibn_cityflowv2_384px_best.pth",
                "checkpoint_dir": "/kaggle/working/checkpoints",
                "backbone": "resnet101_ibn_a",
                "feat_dim": 2048,
                "img_size": (384, 384),
                "gem_p": 3.0,
                "epochs": 120,
                "batch_size": 64,
                "eval_batch_size": 64,
                "num_instances": 4,
                "lr": 3.5e-4,
                "backbone_lr_factor": 0.1,
                "warmup_epochs": 10,
                "eta_min": 1e-6,
                "weight_decay": 1e-4,
                "label_smoothing": 0.1,
                "triplet_margin": 0.3,
                "circle_m": 0.25,
                "circle_gamma": 80,
                "triplet_weight": 1.0,
                "circle_weight": 0.5,
                "id_weight": 1.0,
                "random_erasing_prob": 0.5,
                "color_jitter": True,
                "eval_every": 5,
                "fp16": True,
            }}

            os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
            print(json.dumps(CFG, indent=2, default=str))

            RESUME_FROM = None
            RESUME_EPOCH = 0
            print({{"resume_from": RESUME_FROM, "resume_epoch": RESUME_EPOCH}})
            """
        ).strip(),
        strip_leading_imports(cell_text(original_cells[4])),
        strip_leading_imports(cell_text(original_cells[5])),
        cell_text(original_cells[6]).strip(),
        cell_text(original_cells[7]).strip(),
        build_model_block(),
        cell_text(original_cells[9]).strip(),
        cell_text(original_cells[10]).strip(),
        cell_text(original_cells[11]).strip(),
        cell_text(original_cells[12]).strip(),
        cell_text(original_cells[13]).strip(),
        cell_text(original_cells[14]).strip(),
        cell_text(original_cells[15]).strip(),
    ]

    main_body = "\n\n".join(indent(section, "    ") for section in main_sections if section)

    training_script = "\n\n".join(
        [
            imports_block,
            debug_block,
            "def main():\n" + main_body,
            'if __name__ == "__main__":\n    main()',
        ]
    ) + "\n"

    return sanitize_training_script(training_script)


def build_cell2(training_script: str) -> str:
    return "\n".join(
        [
            "import os",
            "",
            f"script_content = r'''{training_script}'''",
            "",
            f'script_path = "{TRAIN_SCRIPT_PATH}"',
            'with open(script_path, "w", encoding="utf-8") as f:',
            '    f.write(script_content)',
            'print(f"Training script written to {script_path}")',
            'print(f"Script size: {os.path.getsize(script_path) / 1024:.1f} KB")',
        ]
    )


def build_cell3() -> str:
    return dedent(
        f"""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "{TRAIN_SCRIPT_PATH}"],
            check=False,
            text=True,
        )
        print(f"Training process exited with code: {{result.returncode}}")
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {{result.returncode}}")
        """
    ).strip()


def verify_source_lines(notebook: dict) -> None:
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        source = cell.get("source", [])
        if len(source) > 1:
            for line in source[:-1]:
                if not line.endswith("\n"):
                    raise RuntimeError(f"Cell {index} has a non-final source line without trailing newline")
            if source[-1].endswith("\n"):
                raise RuntimeError(f"Cell {index} final source line unexpectedly ends with newline")


def summarize_cells(notebook: dict) -> list[str]:
    summaries: list[str] = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        first_lines = [line.rstrip("\n") for line in cell.get("source", [])[:3]]
        preview = " | ".join(line for line in first_lines if line).strip()
        summaries.append(f"Cell {index}: {preview}")
    return summaries


def print_script_preview(training_script: str, line_count: int = 80) -> None:
    print(f"Embedded script preview (first {line_count} lines):")
    for index, line in enumerate(training_script.splitlines()[:line_count], start=1):
        print(f"{index:03d}: {line}")


def main() -> None:
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    original_cells = notebook.get("cells", [])
    training_script = build_training_script(original_cells)

    notebook["cells"] = [
        {
            "cell_type": "code",
            "metadata": {"language": "python"},
            "execution_count": None,
            "outputs": [],
            "source": to_source(build_cell1()),
        },
        {
            "cell_type": "code",
            "metadata": {"language": "python"},
            "execution_count": None,
            "outputs": [],
            "source": to_source(build_cell2(training_script)),
        },
        {
            "cell_type": "code",
            "metadata": {"language": "python"},
            "execution_count": None,
            "outputs": [],
            "source": to_source(build_cell3()),
        },
    ]

    with NOTEBOOK_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        verified = json.load(handle)

    if len(verified.get("cells", [])) != 3:
        raise RuntimeError(f"Verification failed: expected 3 cells, found {len(verified.get('cells', []))}")

    verify_source_lines(verified)

    for index, cell in enumerate(verified["cells"], start=1):
        compile_python(cell_text(cell), f"{NOTEBOOK_PATH.name}:cell{index}")

    embedded_training_script = extract_training_script_from_rebuilt_notebook(verified["cells"])
    if embedded_training_script is None:
        raise RuntimeError("Verification failed: could not extract embedded training script from rebuilt notebook")
    compile_python(embedded_training_script, TRAIN_SCRIPT_PATH)

    local_bindings = find_main_local_bindings(embedded_training_script)
    if local_bindings:
        raise RuntimeError(
            "Verification failed: main() still has local imports/assignments for tracked names:\n"
            + "\n".join(local_bindings)
        )

    required_markers = [
        "nvidia-smi",
        TRAIN_SCRIPT_PATH,
        "Loading VeRi-776 pretrained weights from",
        "class TeeWriter:",
        "subprocess.run(",
    ]
    dumped = json.dumps(verified)
    for marker in required_markers:
        if marker not in dumped:
            raise RuntimeError(f"Verification failed: missing marker {marker}")

    print(f"Rebuilt notebook: {NOTEBOOK_PATH}")
    print(f"Cell count: {len(verified['cells'])}")
    for summary in summarize_cells(verified):
        print(summary)
    print("Compilation checks: notebook cells OK, embedded script OK")
    print("Tracked local imports/assignments inside main(): none found")
    print_script_preview(embedded_training_script)


if __name__ == "__main__":
    main()