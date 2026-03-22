"""
Semantic Drift Detection — Finding Where Responses Go Off the Rails
───────────────────────────────────────────────────────────────────
Analyzes responses sentence by sentence to detect the "drift point"
where a response transitions from grounded facts to fabrication.

Produces a per-sentence trust map — enabling fine-grained
hallucination localization instead of binary labels.
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticDriftDetector:
    """Detect sentence-level semantic drift in LLM responses.

    Method:
    1. Split response into sentences
    2. For each sentence, compute similarity to reference/context
    3. Track similarity trajectory over the response
    4. Detect "drift point" where similarity drops sharply
    5. Flag post-drift sentences as likely hallucinations
    """

    def __init__(self, drift_threshold: float = 0.3, window_size: int = 2):
        self.drift_threshold = drift_threshold
        self.window_size = window_size

    def analyze_drift_data(self, drift_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze pre-annotated drift data and compute detection metrics."""
        print(f"Semantic drift analysis on {drift_df['query'].nunique()} responses")

        df = drift_df.copy()

        # For each response, compute drift metrics
        results = []
        for query in df["query"].unique():
            response_data = df[df["query"] == query].sort_values("sentence_idx")
            trust_scores = response_data["trust_score"].values

            # Detect drift point from trust score trajectory
            detected_drift = self._detect_drift_point(trust_scores)
            actual_drift = response_data["drift_point"].iloc[0]

            for _, row in response_data.iterrows():
                results.append({
                    "query": row["query"],
                    "sentence_idx": row["sentence_idx"],
                    "sentence": row["sentence"],
                    "trust_score": row["trust_score"],
                    "is_hallucination": row["is_hallucination"],
                    "detected_drift_point": detected_drift,
                    "actual_drift_point": actual_drift,
                    "post_drift": row["sentence_idx"] >= detected_drift if detected_drift is not None else False,
                })

        result_df = pd.DataFrame(results)

        # Evaluate detection accuracy
        if len(result_df) > 0:
            result_df["prediction_correct"] = result_df["post_drift"] == result_df["is_hallucination"]
            accuracy = result_df["prediction_correct"].mean()
            print(f"  Sentence-level drift detection accuracy: {accuracy:.1%}")

        return result_df

    def analyze_response(
        self,
        response: str,
        reference: str = None,
    ) -> list[dict]:
        """Analyze a single response for semantic drift.

        Args:
            response: LLM response text.
            reference: Optional reference text to compare against.

        Returns:
            List of dicts with per-sentence trust scores.
        """
        sentences = self._split_sentences(response)
        if not sentences:
            return []

        if reference:
            # Compute similarity of each sentence to reference
            trust_scores = self._compute_reference_similarity(sentences, reference)
        else:
            # Use internal coherence (similarity between consecutive sentences)
            trust_scores = self._compute_internal_coherence(sentences)

        drift_point = self._detect_drift_point(trust_scores)

        results = []
        for i, (sentence, score) in enumerate(zip(sentences, trust_scores)):
            results.append({
                "sentence_idx": i,
                "sentence": sentence,
                "trust_score": round(score, 3),
                "is_post_drift": i >= drift_point if drift_point is not None else False,
                "drift_point": drift_point,
            })

        return results

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _compute_reference_similarity(
        self, sentences: list[str], reference: str
    ) -> list[float]:
        """Compute cosine similarity of each sentence to reference."""
        try:
            tfidf = TfidfVectorizer(max_features=500, stop_words="english")
            all_texts = [reference] + sentences
            vectors = tfidf.fit_transform(all_texts)
            ref_vector = vectors[0:1]
            sent_vectors = vectors[1:]
            similarities = cosine_similarity(sent_vectors, ref_vector).flatten()
            return similarities.tolist()
        except ValueError:
            return [0.5] * len(sentences)

    def _compute_internal_coherence(self, sentences: list[str]) -> list[float]:
        """Compute coherence based on similarity to previous sentences."""
        if len(sentences) <= 1:
            return [0.8]

        try:
            tfidf = TfidfVectorizer(max_features=500, stop_words="english")
            vectors = tfidf.fit_transform(sentences)
            sim_matrix = cosine_similarity(vectors)

            scores = [0.9]  # First sentence gets high trust
            for i in range(1, len(sentences)):
                # Average similarity to all previous sentences
                prev_sims = sim_matrix[i, :i]
                scores.append(float(prev_sims.mean()))

            return scores
        except ValueError:
            return [0.5] * len(sentences)

    def _detect_drift_point(self, trust_scores: np.ndarray) -> int:
        """Detect the point where trust scores drop sharply."""
        scores = np.array(trust_scores)
        if len(scores) < 3:
            return None

        # Method: find the largest drop between consecutive windows
        max_drop = 0
        drift_idx = None

        for i in range(1, len(scores)):
            # Compare current score to running average of previous scores
            prev_avg = np.mean(scores[max(0, i - self.window_size):i])
            drop = prev_avg - scores[i]

            if drop > max_drop and drop > self.drift_threshold:
                max_drop = drop
                drift_idx = i

        return drift_idx

    def summarize(self, drift_df: pd.DataFrame) -> str:
        if "detected_drift_point" not in drift_df.columns:
            return "No drift data."

        n_queries = drift_df["query"].nunique()
        n_drifted = drift_df.groupby("query")["detected_drift_point"].first().notna().sum()
        accuracy = drift_df["prediction_correct"].mean() if "prediction_correct" in drift_df.columns else 0

        return (
            f"Semantic Drift Detection ({n_queries} responses):\n"
            f"  Drift detected in: {n_drifted}/{n_queries} responses\n"
            f"  Sentence-level accuracy: {accuracy:.1%}"
        )
