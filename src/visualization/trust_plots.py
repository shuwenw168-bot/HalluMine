"""Trust Visualizations for Hallucination Analysis"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3, "font.size": 11, "figure.dpi": 150})


class TrustPlotter:
    def __init__(self, output_dir="results/figures", dpi=300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def _save(self, fig, name):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path}")

    def plot_consistency_distribution(self, df):
        """Distribution of consistency scores colored by hallucination status."""
        if "composite_consistency" not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, color in [("Reliable", "#2ecc71"), ("Hallucinated", "#e74c3c")]:
            is_halluc = label == "Hallucinated"
            subset = df[df["is_hallucination"] == is_halluc]["composite_consistency"].dropna()
            ax.hist(subset, bins=25, alpha=0.6, label=label, color=color, edgecolor="white")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Decision threshold")
        ax.set_xlabel("Composite Consistency Score")
        ax.set_ylabel("Count")
        ax.set_title("Self-Consistency Separates Hallucinations from Reliable Answers")
        ax.legend()
        self._save(fig, "consistency_distribution")

    def plot_risk_heatmap(self, df):
        """Heatmap of hallucination rate by query features."""
        if "question_type" not in df.columns or "is_hallucination" not in df.columns:
            return
        features = ["demands_numeric", "has_temporal_ref", "asks_about_person", "asks_for_list"]
        features = [f for f in features if f in df.columns]
        if not features:
            return

        fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 4))
        if len(features) == 1:
            axes = [axes]

        for ax, feat in zip(axes, features):
            pivot = df.groupby([feat, "is_hallucination"]).size().unstack(fill_value=0)
            if len(pivot.columns) == 2:
                rates = pivot[True] / (pivot[True] + pivot[False])
                rates.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
            ax.set_title(feat.replace("_", " ").title())
            ax.set_ylabel("Hallucination Rate")
            ax.set_ylim(0, 1)
        plt.suptitle("Hallucination Rate by Query Feature", y=1.02)
        plt.tight_layout()
        self._save(fig, "risk_heatmap")

    def plot_faithfulness_comparison(self, rag_df):
        """Compare faithful vs unfaithful RAG responses."""
        if "faithfulness_score" not in rag_df.columns:
            return
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        metrics = ["entity_coverage", "numeric_fidelity", "semantic_similarity"]
        metrics = [m for m in metrics if m in rag_df.columns]

        for ax, metric in zip(axes, metrics):
            for label, color in [("Faithful", "#2ecc71"), ("Unfaithful", "#e74c3c")]:
                is_faithful = label == "Faithful"
                subset = rag_df[rag_df["is_faithful"] == is_faithful][metric]
                ax.hist(subset, bins=15, alpha=0.6, label=label, color=color, edgecolor="white")
            ax.set_title(metric.replace("_", " ").title())
            ax.legend()
        plt.suptitle("RAG Faithfulness Gap: Faithful vs Unfaithful Responses", y=1.02)
        plt.tight_layout()
        self._save(fig, "faithfulness_comparison")

    def plot_drift_trust_map(self, drift_df):
        """Trust score trajectory showing drift points."""
        if "trust_score" not in drift_df.columns:
            return
        queries = drift_df["query"].unique()[:4]
        fig, axes = plt.subplots(len(queries), 1, figsize=(12, 3 * len(queries)))
        if len(queries) == 1:
            axes = [axes]

        for ax, query in zip(axes, queries):
            data = drift_df[drift_df["query"] == query].sort_values("sentence_idx")
            colors = ["#e74c3c" if h else "#2ecc71" for h in data["is_hallucination"]]
            ax.bar(data["sentence_idx"], data["trust_score"], color=colors, edgecolor="white", width=0.8)

            drift_pt = data["drift_point"].iloc[0] if "drift_point" in data.columns else None
            if drift_pt is not None:
                ax.axvline(drift_pt - 0.5, color="orange", linestyle="--", linewidth=2, label=f"Drift point")

            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Sentence Index")
            ax.set_ylabel("Trust Score")
            short_q = query[:60] + "..." if len(query) > 60 else query
            ax.set_title(f'"{short_q}"')
            ax.legend(loc="lower left")

        plt.suptitle("Semantic Drift: Per-Sentence Trust Maps", y=1.01, fontsize=14)
        plt.tight_layout()
        self._save(fig, "drift_trust_maps")

    def plot_consistency_vs_hallucination(self, df):
        """Scatter: consistency score vs hallucination status with ROC-like visualization."""
        if "composite_consistency" not in df.columns:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        thresholds = np.linspace(0, 1, 50)
        tprs, fprs = [], []
        for t in thresholds:
            predicted_halluc = df["composite_consistency"] < t
            actual_halluc = df["is_hallucination"]
            tp = ((predicted_halluc) & (actual_halluc)).sum()
            fp = ((predicted_halluc) & (~actual_halluc)).sum()
            fn = ((~predicted_halluc) & (actual_halluc)).sum()
            tn = ((~predicted_halluc) & (~actual_halluc)).sum()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            tprs.append(tpr)
            fprs.append(fpr)

        ax.plot(fprs, tprs, "b-", linewidth=2, label="Self-Consistency")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC: Consistency Score as Hallucination Detector")
        ax.legend()

        # AUC approximation
        auc = np.abs(np.sum(np.diff(fprs) * np.array(tprs[:-1])))
        ax.text(0.6, 0.2, f"AUC ≈ {auc:.3f}", fontsize=14, fontweight="bold")

        self._save(fig, "consistency_roc")

    def generate_all(self, queries_df=None, rag_df=None, drift_df=None):
        print("\nGenerating visualizations...")
        if queries_df is not None:
            self.plot_consistency_distribution(queries_df)
            self.plot_risk_heatmap(queries_df)
            self.plot_consistency_vs_hallucination(queries_df)
        if rag_df is not None:
            self.plot_faithfulness_comparison(rag_df)
        if drift_df is not None:
            self.plot_drift_trust_map(drift_df)
        print(f"All figures saved to {self.output_dir}/")
