"""
HalluMine: Full Analysis Pipeline
──────────────────────────────────
python experiments/run_full_analysis.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sample_generator import generate_sample_data
from src.features.query_features import QueryFeatureExtractor
from src.miners.consistency_miner import SelfConsistencyMiner
from src.miners.risk_profiler import PromptRiskProfiler
from src.miners.faithfulness_miner import FaithfulnessGapMiner
from src.miners.drift_detector import SemanticDriftDetector
from src.visualization.trust_plots import TrustPlotter


def load_config(path="config/default_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_analysis(config_path="config/default_config.yaml"):
    start = time.time()
    config = load_config(config_path)

    print("=" * 60)
    print("  HalluMine: Predicting LLM Hallucination Patterns")
    print("=" * 60)

    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Data ──
    print("\n[1/6] Generating sample data...")
    data = generate_sample_data(n_total=600, seed=42)
    queries_df = data["queries"]
    consistency_df = data["consistency"]
    rag_df = data["rag"]
    drift_df = data["drift"]

    # ── Step 2: Features ──
    print("\n[2/6] Extracting query features...")
    extractor = QueryFeatureExtractor()
    queries_df = extractor.extract(queries_df)

    # ── Step 3: Self-Consistency Mining ──
    print("\n[3/6] Self-consistency mining...")
    consistency_miner = SelfConsistencyMiner(n_samples=5)
    queries_df = consistency_miner.analyze(queries_df, consistency_df)
    print(consistency_miner.summarize(queries_df))

    # ── Step 4: Prompt Risk Profiling ──
    print("\n[4/6] Prompt risk profiling...")
    profiler = PromptRiskProfiler(
        min_support=config["risk_profiler"]["min_support"],
        min_confidence=config["risk_profiler"]["min_confidence"],
    )
    risk_rules = profiler.fit(queries_df)
    print(profiler.summarize(risk_rules))

    # Add predicted risk scores
    queries_df["predicted_risk"] = profiler.predict_risk(queries_df)

    # ── Step 5: RAG Faithfulness ──
    print("\n[5/6] RAG faithfulness analysis...")
    faith_miner = FaithfulnessGapMiner()
    rag_df = faith_miner.analyze(rag_df)
    print(faith_miner.summarize(rag_df))

    # ── Step 5b: Semantic Drift ──
    print("\n--- Semantic drift detection ---")
    drift_detector = SemanticDriftDetector(
        drift_threshold=config["drift_detection"]["threshold"],
    )
    drift_results = drift_detector.analyze_drift_data(drift_df)
    print(drift_detector.summarize(drift_results))

    # ── Step 6: Visualize & Save ──
    print("\n[6/6] Generating visualizations and saving results...")
    plotter = TrustPlotter(
        output_dir=config["output"]["figures_dir"],
        dpi=config["output"]["figure_dpi"],
    )
    plotter.generate_all(
        queries_df=queries_df,
        rag_df=rag_df,
        drift_df=drift_results if len(drift_results) > 0 else drift_df,
    )

    # Save CSVs
    queries_df.to_csv(results_dir / "query_analysis.csv", index=False)
    profiler.rules_to_dataframe(risk_rules).to_csv(results_dir / "risk_rules.csv", index=False)
    rag_df.to_csv(results_dir / "rag_faithfulness.csv", index=False)
    drift_results.to_csv(results_dir / "drift_analysis.csv", index=False)

    # Summary report
    halluc_rate = queries_df["is_hallucination"].mean()
    consistency_gap = (
        queries_df[~queries_df["is_hallucination"]]["composite_consistency"].mean() -
        queries_df[queries_df["is_hallucination"]]["composite_consistency"].mean()
    )
    rag_acc = (rag_df["predicted_faithful"] == rag_df["is_faithful"]).mean() if "predicted_faithful" in rag_df.columns else 0

    summary = f"""# HalluMine Analysis Report

## Dataset
- Total queries: {len(queries_df)}
- Hallucination rate: {halluc_rate:.1%}
- RAG scenarios: {len(rag_df)}
- Drift examples: {drift_df['query'].nunique()} responses

## Self-Consistency Mining
{consistency_miner.summarize(queries_df)}
- **Consistency gap** (reliable - hallucinated): {consistency_gap:.3f}

## Prompt Risk Profiling
- {len(risk_rules)} risk rules discovered
- Top rules predict hallucination with up to {max(r.confidence for r in risk_rules):.0%} confidence

## RAG Faithfulness
{faith_miner.summarize(rag_df)}
- Detection accuracy: {rag_acc:.1%}

## Semantic Drift
{drift_detector.summarize(drift_results)}

## Key Insight
Self-consistency scores alone separate hallucinated from reliable responses
with a gap of {consistency_gap:.3f} — no external knowledge base required.
This makes it a practical, deployable signal for production systems.
"""
    (results_dir / "analysis_report.md").write_text(summary)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Analysis complete in {elapsed:.1f}s")
    print(f"  Results: {results_dir}/")
    print(f"  Figures: {config['output']['figures_dir']}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_analysis()
