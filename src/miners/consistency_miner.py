"""
Self-Consistency Mining — Predicting Hallucination from Response Variance
─────────────────────────────────────────────────────────────────────────
Key insight: If you ask an LLM the same question 5 times and get 5
different answers, it's probably hallucinating. If you get the same
answer every time, it's probably grounded in real knowledge.

This miner quantifies this intuition with multiple consistency metrics
and mines the patterns of when inconsistency occurs.
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SelfConsistencyMiner:
    """Analyze self-consistency across multiple LLM responses.

    For each query, given N sampled responses:
    1. Compute pairwise semantic similarity (TF-IDF cosine)
    2. Extract answer entities and check agreement
    3. Compute numeric variance (if answers contain numbers)
    4. Produce a composite consistency score

    Low consistency → high hallucination risk.
    """

    def __init__(self, n_samples: int = 5):
        self.n_samples = n_samples

    def analyze(
        self,
        queries_df: pd.DataFrame,
        consistency_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute consistency metrics for each query.

        Args:
            queries_df: DataFrame with prompt_id and query columns.
            consistency_df: DataFrame with prompt_id, sample_idx, response.

        Returns:
            queries_df enriched with consistency metrics.
        """
        print(f"Self-consistency mining on {queries_df['prompt_id'].nunique()} queries "
              f"(×{self.n_samples} samples)")

        results = []
        for pid in queries_df["prompt_id"].unique():
            samples = consistency_df[consistency_df["prompt_id"] == pid]["response"].tolist()
            if len(samples) < 2:
                results.append(self._empty_metrics(pid))
                continue

            metrics = self._compute_metrics(pid, samples)
            results.append(metrics)

        metrics_df = pd.DataFrame(results)
        enriched = queries_df.merge(metrics_df, on="prompt_id", how="left")

        # Mine patterns: what consistency thresholds separate hallucinations?
        if "is_hallucination" in enriched.columns:
            self._report_discrimination(enriched)

        return enriched

    def _compute_metrics(self, prompt_id: int, samples: list[str]) -> dict:
        """Compute all consistency metrics for a set of samples."""
        metrics = {"prompt_id": prompt_id}

        # 1. Semantic consistency (TF-IDF cosine similarity)
        try:
            tfidf = TfidfVectorizer(max_features=500, stop_words="english")
            vectors = tfidf.fit_transform(samples)
            sim_matrix = cosine_similarity(vectors)
            # Average pairwise similarity (excluding diagonal)
            n = sim_matrix.shape[0]
            if n > 1:
                mask = np.ones_like(sim_matrix, dtype=bool)
                np.fill_diagonal(mask, False)
                metrics["semantic_consistency"] = float(sim_matrix[mask].mean())
                metrics["semantic_consistency_min"] = float(sim_matrix[mask].min())
                metrics["semantic_consistency_std"] = float(sim_matrix[mask].std())
            else:
                metrics["semantic_consistency"] = 1.0
                metrics["semantic_consistency_min"] = 1.0
                metrics["semantic_consistency_std"] = 0.0
        except ValueError:
            metrics["semantic_consistency"] = 1.0
            metrics["semantic_consistency_min"] = 1.0
            metrics["semantic_consistency_std"] = 0.0

        # 2. Lexical consistency (exact word overlap)
        word_sets = [set(s.lower().split()) for s in samples]
        if word_sets:
            intersection = word_sets[0]
            union = word_sets[0]
            for ws in word_sets[1:]:
                intersection = intersection & ws
                union = union | ws
            metrics["lexical_overlap"] = len(intersection) / max(len(union), 1)
        else:
            metrics["lexical_overlap"] = 1.0

        # 3. Numeric consistency
        import re
        all_numbers = []
        for s in samples:
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', s)
            all_numbers.append([float(n) for n in numbers])

        if any(all_numbers):
            # Check if the same numbers appear across samples
            flat_numbers = [n for nums in all_numbers for n in nums]
            if flat_numbers:
                metrics["numeric_variance"] = float(np.std(flat_numbers))
                # Count unique first-number across samples
                first_nums = [nums[0] if nums else None for nums in all_numbers]
                first_nums = [n for n in first_nums if n is not None]
                metrics["numeric_agreement"] = (
                    Counter(first_nums).most_common(1)[0][1] / len(first_nums)
                    if first_nums else 1.0
                )
            else:
                metrics["numeric_variance"] = 0.0
                metrics["numeric_agreement"] = 1.0
        else:
            metrics["numeric_variance"] = 0.0
            metrics["numeric_agreement"] = 1.0

        # 4. Length consistency
        lengths = [len(s.split()) for s in samples]
        metrics["length_cv"] = float(np.std(lengths) / max(np.mean(lengths), 1))

        # 5. Composite score (weighted)
        metrics["composite_consistency"] = (
            0.4 * metrics["semantic_consistency"] +
            0.2 * metrics["lexical_overlap"] +
            0.2 * metrics["numeric_agreement"] +
            0.2 * (1.0 - min(metrics["length_cv"], 1.0))
        )

        return metrics

    def _empty_metrics(self, prompt_id: int) -> dict:
        return {
            "prompt_id": prompt_id,
            "semantic_consistency": None,
            "semantic_consistency_min": None,
            "semantic_consistency_std": None,
            "lexical_overlap": None,
            "numeric_variance": None,
            "numeric_agreement": None,
            "length_cv": None,
            "composite_consistency": None,
        }

    def _report_discrimination(self, df: pd.DataFrame):
        """Report how well consistency separates hallucinated from reliable."""
        halluc = df[df["is_hallucination"] == True]["composite_consistency"]
        reliable = df[df["is_hallucination"] == False]["composite_consistency"]

        if len(halluc) > 0 and len(reliable) > 0:
            print(f"\n  Consistency discrimination:")
            print(f"    Hallucinated: mean={halluc.mean():.3f}, std={halluc.std():.3f}")
            print(f"    Reliable:     mean={reliable.mean():.3f}, std={reliable.std():.3f}")
            print(f"    Gap:          {reliable.mean() - halluc.mean():.3f}")

    def get_unreliable_queries(
        self, df: pd.DataFrame, threshold: float = 0.6
    ) -> pd.DataFrame:
        """Return queries with low consistency (likely hallucination)."""
        return df[df["composite_consistency"] < threshold].sort_values(
            "composite_consistency"
        )

    def summarize(self, df: pd.DataFrame) -> str:
        total = len(df)
        if "composite_consistency" not in df.columns:
            return "No consistency data available."

        low = (df["composite_consistency"] < 0.5).sum()
        medium = ((df["composite_consistency"] >= 0.5) & (df["composite_consistency"] < 0.8)).sum()
        high = (df["composite_consistency"] >= 0.8).sum()

        return (
            f"Self-Consistency Analysis ({total} queries):\n"
            f"  High consistency (≥0.8):  {high} ({high/total:.0%}) — likely reliable\n"
            f"  Medium (0.5-0.8):         {medium} ({medium/total:.0%}) — uncertain\n"
            f"  Low (<0.5):               {low} ({low/total:.0%}) — likely hallucinating"
        )
