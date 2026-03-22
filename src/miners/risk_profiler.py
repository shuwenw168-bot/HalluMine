"""
Prompt Risk Profiler — Mining Rules That Predict Hallucination
─────────────────────────────────────────────────────────────
Discovers interpretable rules: {prompt_features} → {hallucination_risk}

Example rules:
  {asks_specific_count, domain=academic} → hallucination (conf=0.78)
  {has_temporal_ref, demands_specificity} → hallucination (conf=0.71)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


@dataclass
class RiskRule:
    """An interpretable rule predicting hallucination risk."""
    conditions: dict[str, str]  # feature → condition
    confidence: float
    support: float
    n_matches: int
    hallucination_rate: float

    def __str__(self):
        conds = " & ".join(f"{k}={v}" for k, v in self.conditions.items())
        return (f"IF {conds} → hallucination_risk "
                f"(conf={self.confidence:.2f}, support={self.support:.2f}, n={self.n_matches})")


class PromptRiskProfiler:
    """Mine interpretable rules that predict which queries will trigger hallucinations.

    Method:
    1. Discretize query features into categorical bins
    2. Mine association rules: feature combinations → hallucination
    3. Also train a decision tree for rule extraction
    4. Rank rules by confidence and interpretability
    """

    FEATURE_COLUMNS = [
        "demands_numeric", "has_temporal_ref", "demands_specificity",
        "asks_about_person", "asks_for_list", "domain_academic",
        "domain_finance", "domain_medical", "question_type",
        "specificity_bin", "length_bin",
    ]

    def __init__(self, min_support: float = 0.05, min_confidence: float = 0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.tree_model = None

    def fit(self, df: pd.DataFrame) -> list[RiskRule]:
        """Mine risk rules from labeled query data."""
        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        print(f"Prompt risk profiling with {len(available)} features on {len(df)} queries")

        rules = []

        # Method 1: Conditional probability mining
        for col in available:
            if df[col].dtype == 'object' or df[col].nunique() <= 5:
                for val in df[col].dropna().unique():
                    subset = df[df[col] == val]
                    if len(subset) < self.min_support * len(df):
                        continue
                    halluc_rate = subset["is_hallucination"].mean()
                    if halluc_rate >= self.min_confidence:
                        rules.append(RiskRule(
                            conditions={col: str(val)},
                            confidence=halluc_rate,
                            support=len(subset) / len(df),
                            n_matches=len(subset),
                            hallucination_rate=halluc_rate,
                        ))

        # Method 2: Pairwise feature combinations
        binary_cols = [c for c in available if df[c].dtype in ['int64', 'float64'] and df[c].nunique() <= 3]
        from itertools import combinations
        for c1, c2 in combinations(binary_cols, 2):
            for v1 in df[c1].unique():
                for v2 in df[c2].unique():
                    subset = df[(df[c1] == v1) & (df[c2] == v2)]
                    if len(subset) < max(self.min_support * len(df), 5):
                        continue
                    halluc_rate = subset["is_hallucination"].mean()
                    if halluc_rate >= self.min_confidence:
                        rules.append(RiskRule(
                            conditions={c1: str(v1), c2: str(v2)},
                            confidence=halluc_rate,
                            support=len(subset) / len(df),
                            n_matches=len(subset),
                            hallucination_rate=halluc_rate,
                        ))

        # Method 3: Decision tree rule extraction
        tree_rules = self._extract_tree_rules(df, available)
        rules.extend(tree_rules)

        # Deduplicate and sort
        rules.sort(key=lambda r: r.confidence, reverse=True)
        print(f"  Found {len(rules)} risk rules (confidence ≥ {self.min_confidence})")
        return rules

    def _extract_tree_rules(self, df: pd.DataFrame, feature_cols: list) -> list[RiskRule]:
        """Train a shallow decision tree and extract rules."""
        # Prepare features (encode categoricals)
        X = pd.get_dummies(df[feature_cols], drop_first=True).fillna(0)
        y = df["is_hallucination"].astype(int)

        self.tree_model = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=max(10, int(0.05 * len(df))),
            class_weight="balanced", random_state=42,
        )
        self.tree_model.fit(X, y)

        # Extract text rules
        tree_text = export_text(self.tree_model, feature_names=list(X.columns), max_depth=3)

        # Parse into RiskRule objects (simplified)
        rules = []
        feature_importances = dict(zip(X.columns, self.tree_model.feature_importances_))
        top_features = sorted(feature_importances.items(), key=lambda x: -x[1])[:5]

        for feat, importance in top_features:
            if importance > 0.05:
                rules.append(RiskRule(
                    conditions={"tree_feature": feat, "importance": f"{importance:.2f}"},
                    confidence=importance,
                    support=0.0,
                    n_matches=0,
                    hallucination_rate=importance,
                ))

        return rules

    def predict_risk(self, df: pd.DataFrame) -> pd.Series:
        """Predict hallucination risk score for new queries."""
        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        X = pd.get_dummies(df[available], drop_first=True).fillna(0)

        if self.tree_model is not None:
            # Align columns
            for col in self.tree_model.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.tree_model.feature_names_in_]
            return pd.Series(self.tree_model.predict_proba(X)[:, 1], index=df.index)
        return pd.Series(0.5, index=df.index)

    def rules_to_dataframe(self, rules: list[RiskRule]) -> pd.DataFrame:
        return pd.DataFrame([{
            "rule": " & ".join(f"{k}={v}" for k, v in r.conditions.items()),
            "confidence": r.confidence,
            "support": r.support,
            "n_matches": r.n_matches,
        } for r in rules])

    def summarize(self, rules: list[RiskRule]) -> str:
        if not rules:
            return "No risk rules found."
        lines = [f"Found {len(rules)} hallucination risk rules:\n"]
        for i, r in enumerate(rules[:8], 1):
            lines.append(f"  {i}. {r}")
        return "\n".join(lines)
