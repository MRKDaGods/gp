"""Convert VS Code XML cell format notebooks to proper Jupyter JSON format."""
import re
import json
import sys
from pathlib import Path


def convert_vscode_xml_to_ipynb(xml_path: str, ipynb_path: str):
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Handle both \r\n and \n line endings
    content = content.replace("\r\n", "\n")

    pattern = r'<VSCode\.Cell\s+id="[^"]*"\s+language="(\w+)">\n(.*?)\n</VSCode\.Cell>'
    matches = re.findall(pattern, content, re.DOTALL)

    cells = []
    for lang, source in matches:
        cell_type = "markdown" if lang == "markdown" else "code"
        lines = source.split("\n")
        # Add newlines to all but last line
        source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []

        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source_lines,
        }
        if cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

        cells.append(cell)

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"Converted {len(cells)} cells to {ipynb_path}")


if __name__ == "__main__":
    convert_vscode_xml_to_ipynb(
        "notebooks/kaggle/07_person_reid_sota/07_person_reid_sota.ipynb",
        "notebooks/kaggle/07_person_reid_sota/07_person_reid_sota_kaggle.ipynb",
    )
    convert_vscode_xml_to_ipynb(
        "notebooks/kaggle/08_vehicle_reid_sota/08_vehicle_reid_sota.ipynb",
        "notebooks/kaggle/08_vehicle_reid_sota/08_vehicle_reid_sota_kaggle.ipynb",
    )
