from __future__ import annotations

import json
from pathlib import Path


KERNEL_METADATA_PATH = Path("notebooks/kaggle/11c_wildtrack_stages45/kernel-metadata.json")
RESTORED_KERNEL_SOURCES = [
    "mrkdagods/mtmc-11b-wildtrack-stage-3-faiss-indexing",
]


def main() -> None:
    data = json.loads(KERNEL_METADATA_PATH.read_text(encoding="utf-8"))
    data["kernel_sources"] = RESTORED_KERNEL_SOURCES
    KERNEL_METADATA_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()