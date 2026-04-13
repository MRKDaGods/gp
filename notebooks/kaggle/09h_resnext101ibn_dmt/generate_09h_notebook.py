import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEMPLATE_NOTEBOOK_PATH = HERE.parent / "09g_resnet101ibn_dmt" / "09g_resnet101ibn_dmt.ipynb"
TEMPLATE_METADATA_PATH = HERE.parent / "09g_resnet101ibn_dmt" / "kernel-metadata.json"
NOTEBOOK_PATH = HERE / "09h_resnext101ibn_dmt.ipynb"
METADATA_PATH = HERE / "kernel-metadata.json"


def to_source(text: str) -> list[str]:
    text = text.strip("\n")
    if not text:
        return []
    lines = text.splitlines()
    return [f"{line}\n" for line in lines[:-1]] + [lines[-1]]


def source_to_text(source: list[str]) -> str:
    return "\n".join(line[:-1] if line.endswith("\n") else line for line in source)


def replace_if_present(text: str, old: str, new: str) -> str:
    if old not in text:
        return text
    return text.replace(old, new)


def transform_cell_source(source: list[str]) -> list[str]:
    text = source_to_text(source)

    replacements = [
        ("SEED = 42", "SEED = 2024"),
        ('OUTPUT_DIR = Path("/kaggle/working/09g_output")', 'OUTPUT_DIR = Path("/kaggle/working/09h_output")'),
        (
            '    "target_map": 0.65,',
            '    "target_map": 0.65,\n    "color_jitter_brightness": 0.25,',
        ),
        (
            '        T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.0),',
            '        T.ColorJitter(brightness=CFG["color_jitter_brightness"], contrast=0.15, saturation=0.1, hue=0.0),',
        ),
        (
            'IBN_NET_URL = "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth"',
            'IBN_NET_URL = "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth"',
        ),
        (
            'IBN_WEIGHTS_PATH = OUTPUT_DIR / "resnet101_ibn_a_imagenet.pth"',
            'IBN_WEIGHTS_PATH = OUTPUT_DIR / "resnext101_ibn_a_imagenet.pth"',
        ),
        ('class ResNet101IBNNeck(nn.Module):', 'class ResNeXt101IBNNeck(nn.Module):'),
        (
            'if hasattr(timm, "list_models") and "resnet101_ibn_a" in timm.list_models(pretrained=False):',
            'if hasattr(timm, "list_models") and "resnext101_32x4d" in timm.list_models(pretrained=False):',
        ),
        (
            'print("timm exposes resnet101_ibn_a, but using explicit IBN-Net patch to preserve last_stride=1 and GeM feature maps")',
            'print("timm exposes resnext101_32x4d, but using explicit IBN-Net patch to preserve last_stride=1 and GeM feature maps")',
        ),
        (
            'base = tv_models.resnet101(weights=None)',
            'from torchvision.models.resnet import ResNet, Bottleneck\n        base = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=4)',
        ),
        (
            'model = ResNet101IBNNeck(num_classes=num_classes, gem_p=CFG["gem_p"], pretrained=True)',
            'model = ResNeXt101IBNNeck(num_classes=num_classes, gem_p=CFG["gem_p"], pretrained=True)',
        ),
        ('"09g_target_mAP": CFG["target_map"],', '"09h_target_mAP": CFG["target_map"],'),
        (
            'BEST_MODEL_PATH = Path("/kaggle/working/resnet101ibn_dmt_best.pth")',
            'BEST_MODEL_PATH = Path("/kaggle/working/resnext101ibn_dmt_best.pth")',
        ),
        (
            'METADATA_PATH = Path("/kaggle/working/resnet101ibn_dmt_metadata.json")',
            'METADATA_PATH = Path("/kaggle/working/resnext101ibn_dmt_metadata.json")',
        ),
        (
            'HISTORY_PATH = Path("/kaggle/working/resnet101ibn_dmt_history.json")',
            'HISTORY_PATH = Path("/kaggle/working/resnext101ibn_dmt_history.json")',
        ),
        (
            '    "model": "resnet101_ibn_a",',
            '    "model": "resnext101_ibn_a",\n    "backbone_variant": "resnext101_32x4d",',
        ),
    ]

    for old, new in replacements:
        text = replace_if_present(text, old, new)

    return to_source(text)


def build_notebook() -> dict:
    with TEMPLATE_NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    cells = payload.get("cells", [])
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["source"] = transform_cell_source(cell.get("source", []))

    notebook_text = "\n\n".join(
        source_to_text(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "code"
    )
    required_markers = [
        'SEED = 2024',
        'Path("/kaggle/working/09h_output")',
        'CFG["color_jitter_brightness"]',
        'resnext101_ibn_a-6ace051d.pth',
        'ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=4)',
        'class ResNeXt101IBNNeck(nn.Module):',
        'Path("/kaggle/working/resnext101ibn_dmt_best.pth")',
        '"model": "resnext101_ibn_a"',
        '"backbone_variant": "resnext101_32x4d"',
        '"09h_target_mAP": CFG["target_map"]',
    ]
    for marker in required_markers:
        if marker not in notebook_text:
            raise RuntimeError(f"Missing required marker after transform: {marker}")

    forbidden_markers = [
        'SEED = 42',
        'Path("/kaggle/working/09g_output")',
        'self.instance_norm',
        'self.batch_norm',
        'class ResNet101IBNNeck(nn.Module):',
        'resnext101_32x8d',
        'Path("/kaggle/working/resnet101ibn_dmt_best.pth")',
        '"09g_target_mAP": CFG["target_map"]',
        '"model": "resnet101_ibn_a"',
    ]
    for marker in forbidden_markers:
        if marker in notebook_text:
            raise RuntimeError(f"Unexpected stale marker after transform: {marker}")

    return payload


def build_metadata() -> dict:
    with TEMPLATE_METADATA_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    payload.update(
        {
            "id": "ali369/09h-resnext101-ibn-a-dmt-cityflowv2",
            "title": "09h ResNeXt101-IBN-a DMT CityFlowV2",
            "code_file": "09h_resnext101ibn_dmt.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "machine_shape": "NvidiaTeslaT4",
            "enable_internet": True,
            "dataset_sources": [
                "mrkdagods/mtmc-weights",
                "thanhnguyenle/data-aicity-2023-track-2",
            ],
            "kernel_sources": [],
            "competition_sources": [],
        }
    )
    return payload


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def main() -> None:
    notebook_payload = build_notebook()
    metadata_payload = build_metadata()
    write_json(NOTEBOOK_PATH, notebook_payload)
    write_json(METADATA_PATH, metadata_payload)
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Wrote {METADATA_PATH}")


if __name__ == "__main__":
    main()