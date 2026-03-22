"""Tests for HalluMine. Run: pytest tests/ -v"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from src.data.sample_generator import generate_sample_data
from src.features.query_features import QueryFeatureExtractor
from src.miners.consistency_miner import SelfConsistencyMiner
from src.miners.risk_profiler import PromptRiskProfiler
from src.miners.faithfulness_miner import FaithfulnessGapMiner
from src.miners.drift_detector import SemanticDriftDetector


@pytest.fixture
def sample_data():
    return generate_sample_data(n_total=200, seed=42)


class TestSampleGenerator:
    def test_returns_all_dataframes(self, sample_data):
        assert "queries" in sample_data
        assert "consistency" in sample_data
        assert "rag" in sample_data
        assert "drift" in sample_data

    def test_has_hallucination_labels(self, sample_data):
        df = sample_data["queries"]
        assert "is_hallucination" in df.columns
        assert df["is_hallucination"].any()  # Some are hallucinated
        assert not df["is_hallucination"].all()  # Not all


class TestConsistencyMiner:
    def test_computes_scores(self, sample_data):
        miner = SelfConsistencyMiner(n_samples=5)
        result = miner.analyze(sample_data["queries"], sample_data["consistency"])
        assert "composite_consistency" in result.columns
        assert result["composite_consistency"].notna().any()

    def test_hallucinated_less_consistent(self, sample_data):
        miner = SelfConsistencyMiner(n_samples=5)
        result = miner.analyze(sample_data["queries"], sample_data["consistency"])
        halluc = result[result["is_hallucination"]]["composite_consistency"].mean()
        reliable = result[~result["is_hallucination"]]["composite_consistency"].mean()
        assert reliable > halluc  # Reliable should be more consistent


class TestRiskProfiler:
    def test_finds_rules(self, sample_data):
        extractor = QueryFeatureExtractor()
        df = extractor.extract(sample_data["queries"])
        miner = SelfConsistencyMiner()
        df = miner.analyze(df, sample_data["consistency"])
        profiler = PromptRiskProfiler(min_confidence=0.3)
        rules = profiler.fit(df)
        assert len(rules) > 0


class TestFaithfulness:
    def test_detects_unfaithful(self, sample_data):
        miner = FaithfulnessGapMiner()
        result = miner.analyze(sample_data["rag"])
        assert "faithfulness_score" in result.columns
        faithful_mean = result[result["is_faithful"]]["faithfulness_score"].mean()
        unfaithful_mean = result[~result["is_faithful"]]["faithfulness_score"].mean()
        assert faithful_mean > unfaithful_mean


class TestDriftDetector:
    def test_detects_drift(self, sample_data):
        detector = SemanticDriftDetector()
        result = detector.analyze_drift_data(sample_data["drift"])
        assert "detected_drift_point" in result.columns
