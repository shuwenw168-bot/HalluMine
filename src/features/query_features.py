"""
Query Feature Extraction for Hallucination Risk Profiling
──────────────────────────────────────────────────────────
Extracts structural and semantic features from queries that
predict hallucination risk. These become the "items" that
association rule mining operates on.
"""

import re

import numpy as np
import pandas as pd


class QueryFeatureExtractor:
    """Extract hallucination-predictive features from queries."""

    # Patterns that correlate with higher hallucination risk
    NUMERIC_PATTERNS = [
        r'how many', r'exactly \d+', r'what (percentage|number|count)',
        r'list \d+', r'name \d+', r'give me \d+',
    ]
    TEMPORAL_PATTERNS = [
        r'in \d{4}', r'last (year|month|week)', r'recent',
        r'this year', r'current', r'latest', r'20\d\d',
    ]
    SPECIFICITY_PATTERNS = [
        r'exactly', r'precisely', r'specific', r'particular',
    ]
    PERSON_PATTERNS = [
        r'\bwho\b', r'\bdr\.?\b', r'\bprofessor\b', r'\bartist\b',
        r'\bauthor\b', r'\bceo\b', r'\bfounder\b',
    ]

    def extract(self, df: pd.DataFrame, text_column: str = "query") -> pd.DataFrame:
        """Add feature columns to the DataFrame."""
        df = df.copy()
        texts = df[text_column].fillna("")

        # Structural features
        df["query_word_count"] = texts.str.split().str.len()
        df["query_char_count"] = texts.str.len()
        df["has_question_mark"] = texts.str.contains(r'\?').astype(int)
        df["question_type"] = texts.apply(self._classify_question_type)

        # Risk-correlated patterns
        df["demands_numeric"] = texts.str.contains(
            "|".join(self.NUMERIC_PATTERNS), case=False, regex=True
        ).astype(int)
        df["has_temporal_ref"] = texts.str.contains(
            "|".join(self.TEMPORAL_PATTERNS), case=False, regex=True
        ).astype(int)
        df["demands_specificity"] = texts.str.contains(
            "|".join(self.SPECIFICITY_PATTERNS), case=False, regex=True
        ).astype(int)
        df["asks_about_person"] = texts.str.contains(
            "|".join(self.PERSON_PATTERNS), case=False, regex=True
        ).astype(int)
        df["asks_for_list"] = texts.str.contains(
            r'\blist\b|\bname\b|\benumerate\b', case=False, regex=True
        ).astype(int)

        # Domain hints
        df["domain_academic"] = texts.str.contains(
            r'paper|publish|research|study|professor', case=False
        ).astype(int)
        df["domain_finance"] = texts.str.contains(
            r'gdp|revenue|stock|market|economy|price', case=False
        ).astype(int)
        df["domain_medical"] = texts.str.contains(
            r'symptom|treatment|disease|patient|medication', case=False
        ).astype(int)

        # Composite risk indicators
        df["specificity_demand_score"] = (
            df["demands_numeric"] +
            df["demands_specificity"] +
            df["asks_for_list"] +
            df["has_temporal_ref"]
        )

        # Discretized bins for association rule mining
        df["length_bin"] = pd.cut(
            df["query_word_count"], bins=[0, 8, 15, 100],
            labels=["short", "medium", "long"]
        )
        df["specificity_bin"] = pd.cut(
            df["specificity_demand_score"], bins=[-1, 0, 1, 10],
            labels=["low", "medium", "high"]
        )

        print(f"Extracted features for {len(df)} queries")
        return df

    def _classify_question_type(self, text: str) -> str:
        text = text.lower().strip()
        if text.startswith("how many") or text.startswith("what number"):
            return "quantity"
        elif text.startswith("who"):
            return "person"
        elif text.startswith("when") or "what year" in text:
            return "temporal"
        elif text.startswith("list") or text.startswith("name"):
            return "enumeration"
        elif text.startswith("what is") or text.startswith("what are"):
            return "definition"
        elif text.startswith("how"):
            return "process"
        elif text.startswith("why"):
            return "explanation"
        else:
            return "other"
