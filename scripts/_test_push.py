"""Minimal test push to Kaggle."""
import subprocess, json, tempfile, os

nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [
        {
            "cell_type": "code",
            "source": ["print('hello')"],
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        }
    ],
}
meta = {
    "id": "yahiaakhalafallah/test-push-10a",
    "title": "Test Push 10a",
    "code_file": "test.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}
d = tempfile.mkdtemp()
with open(os.path.join(d, "test.ipynb"), "w") as f:
    json.dump(nb, f)
with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
r = subprocess.run(["kaggle", "kernels", "push"], cwd=d, capture_output=True, text=True)
print("stdout:", r.stdout)
print("stderr:", r.stderr)
print("returncode:", r.returncode)
