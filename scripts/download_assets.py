"""Download and verify external checkpoints and datasets for local setup.

The repo intentionally does not store model weights or raw datasets. This script
pulls the known public Kaggle artifacts into the local paths used by configs and
verification scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / "tmp_asset_downloads"
CITYFLOWV2_URL = "https://www.aicitychallenge.org/2022-data-and-evaluation/"


@dataclass(frozen=True)
class AssetSpec:
    id: str
    label: str
    group: str
    final_path: Path
    source_kind: str
    source: str | None = None
    member: str | None = None
    expected_size_bytes: int | None = None
    min_size_bytes: int | None = None
    max_size_bytes: int | None = None
    expected_md5: str | None = None
    required_children: tuple[str, ...] = ()
    manual_url: str | None = None
    optional: bool = False

    @property
    def display_path(self) -> str:
        return self.final_path.as_posix()

    @property
    def is_manual(self) -> bool:
        return self.source_kind == "manual"


ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        id="clipsenet_v6",
        label="CLIP-SENet v6 VeRi-776 checkpoint",
        group="reid",
        final_path=Path("models/reid/clipsenet_v6_veri776_best.pth"),
        source_kind="kernel",
        source="yahiaakhalafallah/13-clip-senet-train",
        member="checkpoints/best.pth",
        expected_size_bytes=1_111_738_778,
        expected_md5="a55f55f60d404c7df5cb62690d7213dc",
    ),
    AssetSpec(
        id="transreid_veri776_09v",
        label="09v TransReID VeRi-776 checkpoint",
        group="reid",
        final_path=Path("models/reid/vehicle_transreid_vit_base_veri776.pth"),
        source_kind="dataset",
        source="mrkdagods/mtmc-weights",
        member="reid/vehicle_transreid_vit_base_veri776.pth",
        expected_size_bytes=346_889_637,
        expected_md5="1ddd5b60cf6071a7794be169b41f63e1",
    ),
    AssetSpec(
        id="transreid_cityflowv2",
        label="CityFlowV2 TransReID checkpoint",
        group="reid",
        final_path=Path("models/reid/transreid_cityflowv2_best.pth"),
        source_kind="dataset",
        source="gumfreddy/mtmc-weights",
        member="reid/transreid_cityflowv2_best.pth",
        expected_size_bytes=346_518_635,
        expected_md5="bee2e2bc7e733d8eb3574abcc10ef5ed",
    ),
    AssetSpec(
        id="mvdetr_wildtrack",
        label="MVDeTr WILDTRACK checkpoint",
        group="detection",
        final_path=Path("models/person_detection/MultiviewDetector.pth"),
        source_kind="dataset",
        source="gumfreddy/12a-wildtrack-mvdetr-checkpoint",
        member="MultiviewDetector.pth",
        expected_size_bytes=49_745_811,
        expected_md5="18658027791f44357f07db6b9406b120",
    ),
    AssetSpec(
        id="veri776_dataset",
        label="VeRi-776 evaluation dataset",
        group="dataset",
        final_path=Path("data/raw/veri776"),
        source_kind="dataset_dir",
        source="abhyudaya12/veri-vehicle-re-identification-dataset",
        min_size_bytes=250_000_000,
        required_children=("image_query", "image_test", "name_query.txt", "name_test.txt"),
        optional=True,
    ),
    AssetSpec(
        id="cityflowv2_dataset",
        label="CityFlowV2 dataset",
        group="manual",
        final_path=Path("data/raw/cityflowv2"),
        source_kind="manual",
        min_size_bytes=1,
        manual_url=CITYFLOWV2_URL,
        optional=True,
    ),
)

ASSETS_BY_ID = {asset.id: asset for asset in ASSETS}
REQUIRED_MODEL_ASSET_IDS = (
    "clipsenet_v6",
    "transreid_veri776_09v",
    "transreid_cityflowv2",
    "mvdetr_wildtrack",
)


@dataclass(frozen=True)
class VerificationResult:
    asset: AssetSpec
    ok: bool
    status: str
    size_bytes: int | None = None
    md5: str | None = None


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def format_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    units = ("B", "KB", "MB", "GB", "TB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return 0


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(asset: AssetSpec) -> VerificationResult:
    path = resolve_path(asset.final_path)
    if not path.exists():
        if asset.is_manual:
            return VerificationResult(asset, False, f"manual download required: {asset.manual_url}")
        return VerificationResult(asset, False, "missing")

    for child in asset.required_children:
        if not (path / child).exists():
            return VerificationResult(asset, False, f"missing child: {child}", tree_size(path))

    size = tree_size(path)
    if asset.expected_size_bytes is not None and size != asset.expected_size_bytes:
        return VerificationResult(
            asset,
            False,
            f"size mismatch: expected {asset.expected_size_bytes}, found {size}",
            size,
        )
    if asset.min_size_bytes is not None and size < asset.min_size_bytes:
        return VerificationResult(asset, False, "too small", size)
    if asset.max_size_bytes is not None and size > asset.max_size_bytes:
        return VerificationResult(asset, False, "too large", size)

    md5 = file_md5(path) if path.is_file() else None
    if asset.expected_md5 is not None and md5 != asset.expected_md5:
        return VerificationResult(asset, False, "md5 mismatch", size, md5)

    return VerificationResult(asset, True, "ok", size, md5)


def verify_assets(assets: Iterable[AssetSpec]) -> list[VerificationResult]:
    return [verify_asset(asset) for asset in assets]


def markdown_table(results: Sequence[VerificationResult]) -> str:
    lines = [
        "| Asset | Path | Size | MD5 | Status |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for result in results:
        md5 = result.md5 or result.asset.expected_md5 or "-"
        status = "OK" if result.ok else result.status
        lines.append(
            "| "
            f"{result.asset.label} | `{result.asset.display_path}` | "
            f"{format_bytes(result.size_bytes)} | `{md5}` | {status} |"
        )
    return "\n".join(lines)


def selected_assets(args: argparse.Namespace) -> list[AssetSpec]:
    groups: set[str] = set()
    if args.all:
        groups.update({"reid", "detection", "dataset", "manual"})
    if args.reid_only:
        groups.add("reid")
    if args.detection_only:
        groups.add("detection")
    if args.datasets:
        groups.update({"dataset", "manual"})
    if not groups:
        groups.update({"reid", "detection"})
    return [asset for asset in ASSETS if asset.group in groups]


def command_for(asset: AssetSpec, temp_dir: Path) -> list[str] | None:
    if asset.source_kind in {"dataset", "dataset_dir"}:
        return ["kaggle", "datasets", "download", "-d", asset.source or "", "-p", str(temp_dir), "--unzip"]
    if asset.source_kind == "kernel":
        return ["kaggle", "kernels", "output", asset.source or "", "-p", str(temp_dir)]
    return None


def print_dry_run(asset: AssetSpec) -> None:
    temp_preview = TEMP_ROOT / asset.id
    cmd = command_for(asset, temp_preview)
    print(f"\n[{asset.id}] {asset.label}")
    if asset.is_manual:
        print(f"  Manual: download from {asset.manual_url} and place at {asset.display_path}")
        return
    print(f"  Command: {' '.join(cmd or [])}")
    if asset.member:
        print(f"  Extract: {asset.member} -> {asset.display_path}")
    else:
        print(f"  Extract required children -> {asset.display_path}")


def extract_zip_files(temp_dir: Path) -> None:
    for zip_path in temp_dir.rglob("*.zip"):
        shutil.unpack_archive(str(zip_path), str(zip_path.parent))


def find_member(temp_dir: Path, member: str) -> Path | None:
    exact = temp_dir / member
    if exact.exists():
        return exact
    normalized = Path(member)
    candidates = [path for path in temp_dir.rglob(normalized.name) if path.is_file()]
    for candidate in candidates:
        if candidate.as_posix().endswith(normalized.as_posix()):
            return candidate
    return candidates[0] if candidates else None


def find_dir_with_children(temp_dir: Path, children: Sequence[str]) -> Path | None:
    for candidate in [temp_dir, *[path for path in temp_dir.rglob("*") if path.is_dir()]]:
        if all((candidate / child).exists() for child in children):
            return candidate
    return None


def replace_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def install_downloaded_asset(asset: AssetSpec, temp_dir: Path) -> None:
    destination = resolve_path(asset.final_path)
    extract_zip_files(temp_dir)

    if asset.source_kind == "dataset_dir":
        source_dir = find_dir_with_children(temp_dir, asset.required_children)
        if source_dir is None:
            raise FileNotFoundError(
                f"Could not find {asset.required_children} in downloaded dataset for {asset.id}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        for child in asset.required_children:
            replace_path(source_dir / child, destination / child)
        return

    if not asset.member:
        raise ValueError(f"Asset {asset.id} has no member to install")
    source_file = find_member(temp_dir, asset.member)
    if source_file is None:
        raise FileNotFoundError(f"Could not find {asset.member} in downloaded files for {asset.id}")
    replace_path(source_file, destination)


def run_command(cmd: Sequence[str]) -> bool:
    print(f"  Running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return completed.returncode == 0


def ensure_kaggle_cli() -> bool:
    if shutil.which("kaggle") is not None:
        return True
    print("Kaggle CLI not found. Install requirements and configure ~/.kaggle/kaggle.json.")
    return False


def process_asset(asset: AssetSpec, force: bool, dry_run: bool) -> VerificationResult:
    print(f"\n=== {asset.label} ===")
    if dry_run:
        print_dry_run(asset)
        return verify_asset(asset)
    if asset.is_manual:
        print(f"Manual asset: download CityFlowV2 from {asset.manual_url}")
        return verify_asset(asset)

    current = verify_asset(asset)
    if current.ok and not force:
        print(f"  Exists and verified: {asset.display_path} ({format_bytes(current.size_bytes)})")
        return current

    with tempfile.TemporaryDirectory(prefix=f"{asset.id}_", dir=TEMP_ROOT) as temp_name:
        temp_dir = Path(temp_name)
        cmd = command_for(asset, temp_dir)
        if not cmd:
            return VerificationResult(asset, False, "no download command")
        if not run_command(cmd):
            return VerificationResult(asset, False, "download failed")
        try:
            install_downloaded_asset(asset, temp_dir)
        except Exception as exc:
            return VerificationResult(asset, False, f"install failed: {exc}")

    result = verify_asset(asset)
    print(f"  {result.status}: {asset.display_path} ({format_bytes(result.size_bytes)})")
    return result


def cleanup_temp_root() -> None:
    if TEMP_ROOT.exists() and not any(TEMP_ROOT.iterdir()):
        TEMP_ROOT.rmdir()


def print_disk_usage(paths: Iterable[Path]) -> None:
    total = sum(tree_size(resolve_path(path)) for path in paths if resolve_path(path).exists())
    usage = shutil.disk_usage(REPO_ROOT)
    print(f"\nSelected asset disk usage: {format_bytes(total)}")
    print(f"Repository volume free space: {format_bytes(usage.free)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MTMC Tracker checkpoints and datasets")
    parser.add_argument("--all", action="store_true", help="Download ReID, detection, and datasets")
    parser.add_argument("--reid-only", action="store_true", help="Download only ReID checkpoints")
    parser.add_argument("--detection-only", action="store_true", help="Download only detector checkpoints")
    parser.add_argument("--datasets", action="store_true", help="Download optional public datasets")
    parser.add_argument("--force", action="store_true", help="Re-download assets even if they verify")
    parser.add_argument("--dry-run", action="store_true", help="Show intended actions without downloading")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    assets = selected_assets(args)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Dry run: no files will be downloaded or changed.")
    elif any(not asset.is_manual for asset in assets) and not ensure_kaggle_cli():
        return 2

    results: list[VerificationResult] = []
    try:
        for asset in assets:
            try:
                results.append(process_asset(asset, force=args.force, dry_run=args.dry_run))
            except Exception as exc:  # keep the batch resilient
                results.append(VerificationResult(asset, False, f"error: {exc}"))
                print(f"  ERROR: {exc}")
    finally:
        cleanup_temp_root()

    print("\nDownload summary")
    print(markdown_table(results))
    print_disk_usage(asset.final_path for asset in assets)

    if args.dry_run:
        return 0

    failed_required = [
        result
        for result in results
        if not result.ok and not result.asset.optional and not result.asset.is_manual
    ]
    if failed_required:
        print("\nSome required model assets failed. Re-run after fixing the errors above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())