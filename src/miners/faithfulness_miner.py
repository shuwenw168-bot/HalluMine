"""
Retrieval Faithfulness Gap Miner (RAG)
──────────────────────────────────────
Detects when a RAG model ignores retrieved context and fabricates answers.

Methods:
1. Entity coverage — Are key entities from context present in response?
2. Numeric fidelity — Do numbers match between context and response?
3. Semantic entailment — Does the response follow from the context?
4. Contradiction detection — Does the response contradict the context?
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FaithfulnessGapMiner:
    """Detect when RAG responses diverge from retrieved context."""

    def analyze(self, rag_df: pd.DataFrame) -> pd.DataFrame:
        """Compute faithfulness metrics for each RAG response.

        Args:
            rag_df: DataFrame with 'context', 'response', and 'query' columns.

        Returns:
            DataFrame enriched with faithfulness metrics.
        """
        print(f"Analyzing RAG faithfulness for {len(rag_df)} responses")
        df = rag_df.copy()

        metrics = [
            self._compute_metrics(row["context"], row["response"], row["query"])
            for _, row in df.iterrows()
        ]
        metrics_df = pd.DataFrame(metrics)
        df = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

        # Composite faithfulness score
        df["faithfulness_score"] = (
            0.3 * df["entity_coverage"] +
            0.2 * df["numeric_fidelity"] +
            0.3 * df["semantic_similarity"] +
            0.2 * (1.0 - df["contradiction_score"])
        )

        df["predicted_faithful"] = df["faithfulness_score"] > 0.5

        if "is_faithful" in df.columns:
            accuracy = (df["predicted_faithful"] == df["is_faithful"]).mean()
            print(f"  Faithfulness detection accuracy: {accuracy:.1%}")

        return df

    def _compute_metrics(self, context: str, response: str, query: str) -> dict:
        """Compute all faithfulness metrics for a single example."""
        context_lower = context.lower()
        response_lower = response.lower()

        # 1. Entity coverage
        context_entities = self._extract_entities(context)
        response_entities = self._extract_entities(response)
        if context_entities:
            entity_coverage = len(context_entities & response_entities) / len(context_entities)
        else:
            entity_coverage = 1.0

        # 2. Numeric fidelity
        context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', context))
        response_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', response))
        if context_numbers:
            numeric_fidelity = len(context_numbers & response_numbers) / len(context_numbers)
        else:
            numeric_fidelity = 1.0

        # 3. Semantic similarity (TF-IDF cosine)
        try:
            tfidf = TfidfVectorizer(max_features=500, stop_words="english")
            vectors = tfidf.fit_transform([context, response])
            semantic_similarity = float(cosine_similarity(vectors[0:1], vectors[1:2])[0, 0])
        except ValueError:
            semantic_similarity = 0.5

        # 4. Contradiction signals
        contradiction_score = self._detect_contradictions(context_lower, response_lower)

        # 5. Fabrication signals (response claims not in context)
        response_claims = set(response_lower.split()) - set("the a an is are was were".split())
        context_words = set(context_lower.split())
        novel_ratio = len(response_claims - context_words) / max(len(response_claims), 1)

        return {
            "entity_coverage": round(entity_coverage, 3),
            "numeric_fidelity": round(numeric_fidelity, 3),
            "semantic_similarity": round(semantic_similarity, 3),
            "contradiction_score": round(contradiction_score, 3),
            "novel_word_ratio": round(novel_ratio, 3),
        }

    def _extract_entities(self, text: str) -> set:
        """Extract named entities (capitalized phrases) and key nouns."""
        # Simple NER via capitalization pattern
        entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        # Also extract numbers as "entities"
        entities.update(re.findall(r'\$?[\d,]+(?:\.\d+)?%?', text))
        return entities

    def _detect_contradictions(self, context: str, response: str) -> float:
        """Detect potential contradictions between context and response."""
        score = 0.0

        # Check for negation flips
        context_negations = len(re.findall(r'\bnot\b|\bno\b|\bnever\b|\bnone\b', context))
        response_negations = len(re.findall(r'\bnot\b|\bno\b|\bnever\b|\bnone\b', response))
        if abs(context_negations - response_negations) > 1:
            score += 0.3

        # Check for numeric disagreements
        context_nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', context)
        response_nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        if context_nums and response_nums:
            c_set = set(float(n) for n in context_nums)
            r_set = set(float(n) for n in response_nums)
            # Numbers in response not in context
            novel_nums = r_set - c_set
            if novel_nums and len(novel_nums) > len(c_set) * 0.5:
                score += 0.4

        # Check for qualifier changes ("all" → "some", etc.)
        amplifiers_context = len(re.findall(r'\ball\b|\bevery\b|\balways\b', context))
        hedges_response = len(re.findall(r'\bsome\b|\bfew\b|\brarely\b', response))
        if amplifiers_context > 0 and hedges_response > 0:
            score += 0.15

        return min(score, 1.0)

    def get_unfaithful(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Return responses flagged as unfaithful."""
        return df[df["faithfulness_score"] < threshold].sort_values("faithfulness_score")

    def summarize(self, df: pd.DataFrame) -> str:
        if "faithfulness_score" not in df.columns:
            return "No faithfulness data."
        faithful = (df["faithfulness_score"] >= 0.5).sum()
        total = len(df)
        return (
            f"RAG Faithfulness Analysis ({total} responses):\n"
            f"  Faithful:   {faithful} ({faithful/total:.0%})\n"
            f"  Unfaithful: {total-faithful} ({(total-faithful)/total:.0%})\n"
            f"  Mean score: {df['faithfulness_score'].mean():.3f}"
        )
