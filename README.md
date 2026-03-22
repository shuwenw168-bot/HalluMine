# HalluMine ⛏️🔮

**Predicting When LLMs Will Hallucinate — Before They Do**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## The Problem

Every hallucination detection tool asks the same question: *"Is this output a hallucination?"*

That's the wrong question. By the time you detect it, you've already served it to the user, wasted compute, and eroded trust.

**HalluMine asks a different question: *"Will this query trigger a hallucination?"***

## The Insight

LLMs don't hallucinate randomly. They hallucinate in **patterns** — certain prompt structures, topic domains, and query types systematically trigger higher hallucination rates. These patterns are mineable.

HalluMine uses four data mining methods to discover and exploit these patterns:

| Method | What It Finds | Why It Matters |
|--------|--------------|----------------|
| **Self-Consistency Mining** | Questions where the LLM gives contradictory answers across samples | High variance = the model is "guessing," not "knowing" |
| **Prompt Risk Profiling** | Structural features of prompts that predict hallucination | Lets you flag risky queries before generation |
| **Retrieval Faithfulness Gap** | When RAG models ignore retrieved context and fabricate answers | The #1 failure mode in production RAG systems |
| **Semantic Drift Detection** | The exact sentence where a response goes from factual to fabricated | Enables fine-grained, sentence-level trust scores |

## Quick Start

```bash
git clone https://github.com/shuwenw168-bot/hallumine.git
cd hallumine
pip install numpy pandas scipy scikit-learn matplotlib seaborn pyyaml tqdm rich
python experiments/run_full_analysis.py
```

No GPU. No API keys. Runs in under 10 seconds.

## Architecture

```
             Query
               │
               ▼
┌──────────────────────────────┐
│      Prompt Risk Profiler    │  "Will this query cause trouble?"
│  (feature extraction +       │
│   association rule mining)   │──── Risk Score: 0.0 - 1.0
└──────────────┬───────────────┘
               │ if high-risk ──→ flag / rephrase / add retrieval
               ▼
┌──────────────────────────────┐
│    LLM Generation (N times)  │  Sample multiple responses
└──────────────┬───────────────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
┌──────────┐     ┌──────────────┐
│  Self-   │     │  Retrieval   │
│Consistency│     │ Faithfulness │  "Did it use the context?"
│  Mining   │     │    Gap       │
└────┬─────┘     └──────┬───────┘
     │                  │
     ▼                  ▼
┌──────────────────────────────┐
│    Semantic Drift Detector   │  "Which sentence went off the rails?"
│  (sentence-level trust map)  │
└──────────────────────────────┘
               │
               ▼
         Trust Report
    (per-sentence scores +
     hallucination risk map)
```

## What Makes This Different

| Existing Tools | HalluMine |
|---|---|
| Detect hallucinations **after** generation | **Predict** hallucination risk **before** generation |
| Binary: hallucinated or not | Continuous risk scores + sentence-level trust maps |
| Need ground truth / knowledge base | Self-consistency requires **no external knowledge** |
| Focus on single responses | Mine **patterns across thousands** of query-response pairs |
| Black-box classifiers | Interpretable rules: "queries with X feature → Y% hallucination rate" |

## Mining Methods

### 1. Self-Consistency Mining

Ask the same question N times. If the LLM gives the same answer every time → it "knows." If answers vary wildly → it's "guessing."

```python
from src.miners.consistency_miner import SelfConsistencyMiner

miner = SelfConsistencyMiner(n_samples=5)
results = miner.analyze(queries_df)

# Output:
# "What year was the Eiffel Tower built?" → consistency=0.98 (reliable)
# "Who was the 7th person on the moon?"   → consistency=0.12 (hallucinating)
```

### 2. Prompt Risk Profiling

Mines association rules: `{prompt_features} → {hallucination_risk}`

```python
from src.miners.risk_profiler import PromptRiskProfiler

profiler = PromptRiskProfiler()
profiler.fit(labeled_df)
rules = profiler.get_risk_rules()

# Example rules:
# {has_numeric_constraint, asks_about_person} → high_risk (confidence=0.78)
# {asks_for_list, specific_count > 5}        → high_risk (confidence=0.71)
```

### 3. Retrieval Faithfulness Gap (RAG)

Detects when a RAG model ignores its retrieved context.

```python
from src.miners.faithfulness_miner import FaithfulnessGapMiner

miner = FaithfulnessGapMiner()
gaps = miner.detect(queries_df, contexts_df, responses_df)

# Flags responses where:
# - Key entities from context are missing in response
# - Response contains claims not grounded in any retrieved passage
# - Numerical values differ between context and response
```

### 4. Semantic Drift Detection

Finds the exact sentence where a response transitions from grounded to fabricated.

```python
from src.miners.drift_detector import SemanticDriftDetector

detector = SemanticDriftDetector()
trust_map = detector.analyze(response_text, reference_text)

# Returns per-sentence scores:
# Sentence 1: 0.95 (grounded)
# Sentence 2: 0.91 (grounded)
# Sentence 3: 0.43 (drifting!)    ← drift point detected
# Sentence 4: 0.21 (fabricated)
```

## Project Structure

```
hallumine/
├── src/
│   ├── data/
│   │   └── sample_generator.py       # Synthetic data with hallucination patterns
│   ├── features/
│   │   └── query_features.py         # Prompt structural feature extraction
│   ├── miners/
│   │   ├── consistency_miner.py      # Self-consistency analysis
│   │   ├── risk_profiler.py          # Prompt → hallucination risk rules
│   │   ├── faithfulness_miner.py     # RAG retrieval faithfulness gap
│   │   └── drift_detector.py         # Semantic drift detection
│   ├── detection/
│   │   └── risk_scorer.py            # Unified hallucination risk scoring
│   └── visualization/
│       └── trust_plots.py            # Trust maps and risk visualizations
├── experiments/
│   └── run_full_analysis.py
├── config/default_config.yaml
├── tests/test_miners.py
└── results/
```

## Use Cases

- **RAG pipeline monitoring** — Flag queries that are likely to produce hallucinated answers before they reach users
- **LLM evaluation** — Systematically discover which topic domains a model is unreliable on
- **Prompt engineering** — Understand which query structures trigger hallucinations so you can design safer prompts
- **Model comparison** — Compare hallucination patterns across models (GPT-4 vs Claude vs Llama)

## Citation

```bibtex
@software{hallumine_2026,
  title={HalluMine: Predicting When LLMs Will Hallucinate Using Data Mining Methods},
  author={[Shuwen Wang]},
  year={2026},
  url={https://github.com/shuwenw168-bot/hallumine}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
