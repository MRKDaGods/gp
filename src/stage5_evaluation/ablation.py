"""Ablation study runner for systematic experiment comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.core.data_models import EvaluationResult


class AblationRunner:
    """Runs the pipeline with different config variants and collects metrics.

    Each variant is a dict of config overrides (dotlist format).
    Results are collected into a DataFrame for comparison.
    """

    def __init__(self, base_config: DictConfig):
        self.base_config = base_config
        self.results: List[Dict] = []

    def add_result(self, variant_name: str, result: EvaluationResult) -> None:
        """Record an evaluation result for a variant."""
        self.results.append({
            "variant": variant_name,
            "mota": result.mota,
            "idf1": result.idf1,
            "hota": result.hota,
            "id_switches": result.id_switches,
            "mostly_tracked": result.mostly_tracked,
            "mostly_lost": result.mostly_lost,
            **{f"detail_{k}": v for k, v in result.details.items()},
        })

    def get_results_table(self) -> pd.DataFrame:
        """Get results as a pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results).set_index("variant")

    def save_results(self, path: str | Path) -> None:
        """Save results table to CSV."""
        df = self.get_results_table()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(path))
        logger.info(f"Ablation results saved to {path}")

    @staticmethod
    def get_standard_variants() -> Dict[str, List[str]]:
        """Get standard ablation study config variants.

        Returns:
            Dict[variant_name, list_of_config_overrides].
        """
        return {
            "baseline": [],
            "tracker_deepocsort": ["stage1.tracker.algorithm=deepocsort"],
            "tracker_bytetrack": ["stage1.tracker.algorithm=bytetrack"],
            "no_reranking": ["stage4.association.reranking.enabled=false"],
            "no_hsv": ["stage4.association.weights.hsv=0.0",
                        "stage4.association.weights.appearance=0.8",
                        "stage4.association.weights.spatiotemporal=0.2"],
            "no_spatiotemporal": ["stage4.association.weights.spatiotemporal=0.0",
                                   "stage4.association.weights.appearance=0.8",
                                   "stage4.association.weights.hsv=0.2"],
            "no_pca": ["stage2.pca.enabled=false"],
            "threshold_0.3": ["stage4.association.graph.similarity_threshold=0.3"],
            "threshold_0.7": ["stage4.association.graph.similarity_threshold=0.7"],
        }
