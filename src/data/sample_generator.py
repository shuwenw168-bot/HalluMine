"""
Sample Data Generator — Realistic Hallucination Patterns
─────────────────────────────────────────────────────────
Generates synthetic query-response data with calibrated hallucination
patterns based on known LLM failure modes:
  - Numeric fabrication (dates, statistics, counts)
  - Entity confusion (mixing up similar people/places)
  - Confident confabulation on obscure topics
  - Faithful responses on well-known topics
  - RAG context ignoring
"""

import random
import re

import numpy as np
import pandas as pd


# ── Query Templates by Risk Level ──

RELIABLE_QUERIES = {
    "common_knowledge": [
        {"query": "What is the capital of France?", "answer": "The capital of France is Paris.", "domain": "geography"},
        {"query": "What is photosynthesis?", "answer": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.", "domain": "science"},
        {"query": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare wrote Romeo and Juliet.", "domain": "literature"},
        {"query": "What is the boiling point of water?", "answer": "Water boils at 100 degrees Celsius or 212 degrees Fahrenheit at standard pressure.", "domain": "science"},
        {"query": "What is the largest planet in our solar system?", "answer": "Jupiter is the largest planet in our solar system.", "domain": "science"},
        {"query": "What language is spoken in Brazil?", "answer": "Portuguese is the official language of Brazil.", "domain": "geography"},
        {"query": "What is the chemical formula for water?", "answer": "The chemical formula for water is H2O.", "domain": "science"},
        {"query": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci painted the Mona Lisa.", "domain": "art"},
    ],
}

RISKY_QUERIES = {
    "numeric_precision": [
        {"query": "How many papers did Geoffrey Hinton publish in 2019?", "true_answer": "approximately 15-20", "halluc_answer": "Geoffrey Hinton published exactly 47 papers in 2019, including groundbreaking work on capsule networks.", "domain": "academic", "risk_type": "numeric_fabrication"},
        {"query": "What was the exact GDP of Portugal in 2018?", "true_answer": "approximately $240 billion", "halluc_answer": "Portugal's GDP in 2018 was exactly $237,842,561,000, making it the 47th largest economy.", "domain": "economics", "risk_type": "numeric_fabrication"},
        {"query": "How many species of beetles exist?", "true_answer": "approximately 400,000 known species", "halluc_answer": "There are exactly 427,318 documented species of beetles as of the latest taxonomic survey.", "domain": "biology", "risk_type": "numeric_fabrication"},
        {"query": "What percentage of the ocean has been explored?", "true_answer": "approximately 5-20% depending on definition", "halluc_answer": "As of 2024, precisely 7.3% of the ocean floor has been mapped in high resolution.", "domain": "science", "risk_type": "numeric_fabrication"},
        {"query": "How many words are in the English language?", "true_answer": "estimates vary, roughly 170,000-250,000 in active use", "halluc_answer": "The Oxford English Dictionary contains exactly 273,149 headwords as of its latest edition.", "domain": "linguistics", "risk_type": "numeric_fabrication"},
    ],
    "obscure_person": [
        {"query": "What awards did Dr. Maria Vasquez receive for her work in computational biology?", "true_answer": "This person may not exist or is not well-known enough to verify", "halluc_answer": "Dr. Maria Vasquez received the ACM Gordon Bell Prize in 2017 and the ISCB Innovator Award in 2019 for her pioneering work in protein folding.", "domain": "academic", "risk_type": "entity_fabrication"},
        {"query": "What is the publication record of Professor James Chen at MIT?", "true_answer": "Cannot verify without more specific identification", "halluc_answer": "Professor James Chen has published 142 papers in top-tier venues, with an h-index of 56, focusing on reinforcement learning.", "domain": "academic", "risk_type": "entity_fabrication"},
        {"query": "Tell me about the artist Kenji Nakamura's recent exhibitions.", "true_answer": "Cannot verify this specific individual", "halluc_answer": "Kenji Nakamura had major exhibitions at the Tate Modern in 2022 and the Guggenheim in 2023, showcasing his neo-digital impressionism.", "domain": "art", "risk_type": "entity_fabrication"},
    ],
    "specific_list": [
        {"query": "List exactly 7 companies that went public in March 2023.", "true_answer": "Would need to verify each specific company", "halluc_answer": "Seven companies that went public in March 2023: TechVista Inc., GreenFlow Energy, DataNova Systems, MediBridge Health, CyberShield Corp, AquaPure Technologies, and NovaStar Robotics.", "domain": "finance", "risk_type": "list_fabrication"},
        {"query": "Name 5 papers published in NeurIPS 2023 about graph neural networks.", "true_answer": "Would need to verify specific paper titles", "halluc_answer": "Five NeurIPS 2023 GNN papers: 'GraphSAGE++: Scaling to Billion Nodes', 'Temporal Graph Transformers for Dynamic Networks', 'Equivariant Message Passing on Heterogeneous Graphs', 'Spectral GNNs with Adaptive Filters', and 'Graph Foundation Models: A Unified Framework'.", "domain": "academic", "risk_type": "list_fabrication"},
    ],
    "temporal_confusion": [
        {"query": "What happened at the UN Climate Summit in November 2024?", "true_answer": "Depends on knowledge cutoff", "halluc_answer": "The November 2024 UN Climate Summit in Dubai resulted in a landmark agreement to phase out fossil fuels by 2050, with 192 nations signing the accord.", "domain": "politics", "risk_type": "temporal_fabrication"},
        {"query": "Who won the 2025 Nobel Prize in Physics?", "true_answer": "May not be known yet", "halluc_answer": "The 2025 Nobel Prize in Physics was awarded to Dr. Sarah Kim and Dr. Raj Patel for their work on quantum error correction.", "domain": "science", "risk_type": "temporal_fabrication"},
    ],
}

# ── RAG Context Scenarios ──
RAG_SCENARIOS = [
    {
        "query": "What is the company's return policy?",
        "context": "Our return policy allows returns within 30 days of purchase with a valid receipt. Items must be in original packaging. Electronics have a 15-day return window. Refunds are processed within 5-7 business days.",
        "faithful_response": "The company allows returns within 30 days with a valid receipt. Items need original packaging. Electronics have a shorter 15-day window. Refunds take 5-7 business days.",
        "unfaithful_response": "The company offers a generous 60-day return policy. All items can be returned regardless of condition. Refunds are typically instant and credited to your original payment method.",
        "domain": "customer_service",
    },
    {
        "query": "What were the Q3 financial results?",
        "context": "Q3 2024 results: Revenue was $4.2B, up 12% YoY. Operating margin improved to 28.3%. Net income was $890M. The company acquired CloudTech for $340M. Headcount grew to 15,200.",
        "faithful_response": "In Q3 2024, the company reported revenue of $4.2 billion, representing 12% year-over-year growth. Operating margin reached 28.3%, and net income was $890 million. The company also acquired CloudTech for $340 million.",
        "unfaithful_response": "Q3 2024 was a record quarter with revenue of $5.1 billion, up 18% year-over-year. The company reported a net income of $1.2 billion and announced plans to acquire three major competitors.",
        "domain": "finance",
    },
    {
        "query": "What are the side effects of the medication?",
        "context": "Common side effects include headache (12% of patients), nausea (8%), and dizziness (5%). Rare but serious side effects include liver damage (0.1%) and allergic reactions (0.3%). The medication should not be taken with alcohol.",
        "faithful_response": "Common side effects are headache (12%), nausea (8%), and dizziness (5%). Rare serious effects include liver damage (0.1%) and allergic reactions (0.3%). Avoid taking with alcohol.",
        "unfaithful_response": "The medication is generally well-tolerated with minimal side effects. Some patients report mild headaches. There are no known serious side effects, and the medication can be safely combined with other substances.",
        "domain": "medical",
    },
    {
        "query": "What does the research paper conclude?",
        "context": "The study found that transformer models trained on code show a 23% improvement in reasoning tasks compared to text-only models (p<0.01, n=1,247). However, this improvement diminished to 8% on tasks requiring common sense reasoning. The authors note limitations in dataset diversity.",
        "faithful_response": "The research found that code-trained transformers improve reasoning performance by 23% over text-only models, though this drops to 8% for common sense reasoning. The study used 1,247 samples and acknowledged limited dataset diversity.",
        "unfaithful_response": "The groundbreaking study conclusively proved that training on code makes LLMs universally better at all reasoning tasks, with improvements of up to 50%. The researchers recommend all future models be trained primarily on code.",
        "domain": "academic",
    },
    {
        "query": "Summarize the meeting notes.",
        "context": "Meeting notes 3/15: Attendees - Sarah, Mike, Priya. Decisions: Launch delayed from April 1 to April 15 due to testing issues. Budget increased by $50K for QA. Mike to lead security review by March 25. Next meeting March 22.",
        "faithful_response": "The March 15 meeting with Sarah, Mike, and Priya decided to delay launch from April 1 to April 15 for additional testing. QA budget was increased by $50K. Mike will complete the security review by March 25. Next meeting is March 22.",
        "unfaithful_response": "The meeting discussed moving the launch date forward to March 25 to meet investor expectations. The team agreed to reduce the QA budget to save costs. Sarah will take over the security review, and the project is on track for early delivery.",
        "domain": "business",
    },
]


# ── Semantic Drift Patterns ──
DRIFT_RESPONSES = [
    {
        "query": "Explain how CRISPR gene editing works.",
        "sentences": [
            ("CRISPR-Cas9 is a gene editing tool adapted from a natural defense mechanism found in bacteria.", 0.95, False),
            ("It uses a guide RNA to direct the Cas9 protein to a specific location in the genome.", 0.93, False),
            ("Once there, Cas9 cuts both strands of the DNA at the target location.", 0.91, False),
            ("The cell's natural repair mechanisms then fix the break, allowing researchers to modify the gene.", 0.88, False),
            ("This technology was first demonstrated in human cells in 2012 by Jennifer Doudna and Emmanuelle Charpentier.", 0.72, False),
            ("Since then, CRISPR has been used to cure sickle cell disease in over 50,000 patients worldwide.", 0.25, True),
            ("The FDA approved CRISPR-based therapies for all major genetic diseases in 2023.", 0.15, True),
        ],
        "drift_point": 5,
    },
    {
        "query": "What is the history of Bitcoin?",
        "sentences": [
            ("Bitcoin was introduced in 2008 through a whitepaper published by the pseudonymous Satoshi Nakamoto.", 0.96, False),
            ("The first Bitcoin transaction occurred in January 2009 when the genesis block was mined.", 0.94, False),
            ("Early Bitcoin had negligible monetary value and was primarily used by cryptography enthusiasts.", 0.90, False),
            ("The first known commercial Bitcoin transaction was in May 2010, when 10,000 BTC were used to buy two pizzas.", 0.92, False),
            ("Bitcoin reached a peak price of approximately $69,000 in November 2021.", 0.85, False),
            ("After the crash, Satoshi Nakamoto revealed his identity as a Japanese-American programmer named Craig Tanaka.", 0.12, True),
            ("He donated all remaining Bitcoin holdings to the United Nations Climate Fund in 2023.", 0.08, True),
        ],
        "drift_point": 5,
    },
    {
        "query": "How does the immune system fight viruses?",
        "sentences": [
            ("When a virus enters the body, the innate immune system provides the first line of defense.", 0.95, False),
            ("Physical barriers like skin and mucous membranes block most pathogens from entering.", 0.94, False),
            ("If a virus breaches these barriers, white blood cells like macrophages and natural killer cells attack it.", 0.92, False),
            ("The adaptive immune system then produces antibodies specific to the virus.", 0.90, False),
            ("T-cells help coordinate the immune response and directly kill infected cells.", 0.88, False),
            ("Memory B-cells retain information about the virus for future protection.", 0.85, False),
            ("This is why most people develop a permanent immunity after recovering from a viral infection, making reinfection impossible.", 0.35, True),
        ],
        "drift_point": 6,
    },
]


def generate_sample_data(n_total: int = 600, seed: int = 42) -> dict:
    """Generate comprehensive synthetic hallucination data.

    Returns a dict with multiple DataFrames for different mining tasks:
      - queries_df: All queries with features and hallucination labels
      - consistency_df: Multiple responses per query for consistency mining
      - rag_df: RAG scenarios with context, faithful/unfaithful responses
      - drift_df: Responses with sentence-level trust scores
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # ── 1. Build Query-Response Pairs ──
    query_rows = []
    consistency_rows = []
    pid = 0

    # Reliable queries (low hallucination risk)
    for _ in range(n_total // 3):
        item = rng.choice(RELIABLE_QUERIES["common_knowledge"])
        query = item["query"]
        answer = item["answer"]

        # Self-consistency: reliable queries get consistent answers
        n_samples = 5
        answers = []
        for s in range(n_samples):
            # Small variations but same core answer
            variation = answer
            if rng.random() < 0.2:
                variation = answer.rstrip(".") + ", which is well-established."
            if rng.random() < 0.1:
                variation = "That's a great question. " + answer
            answers.append(variation)

        # Compute consistency metrics
        consistency = 1.0 - np_rng.uniform(0.0, 0.08)  # Very consistent

        query_rows.append({
            "prompt_id": pid,
            "query": query,
            "response": answer,
            "domain": item["domain"],
            "is_hallucination": False,
            "hallucination_type": "none",
            "risk_level": "low",
            "consistency_score": round(consistency, 3),
            "has_numeric_claim": bool(re.search(r'\d+', answer)),
            "has_named_entity": bool(re.search(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', answer)),
            "asks_about_person": "who" in query.lower(),
            "asks_for_list": any(w in query.lower() for w in ["list", "name", "enumerate"]),
            "asks_specific_count": bool(re.search(r'how many|exactly \d+|list \d+', query.lower())),
            "query_length": len(query.split()),
            "response_length": len(answer.split()),
        })

        for s, ans in enumerate(answers):
            consistency_rows.append({
                "prompt_id": pid, "sample_idx": s, "response": ans,
            })
        pid += 1

    # Risky queries (high hallucination rate)
    for risk_category, items in RISKY_QUERIES.items():
        for _ in range(n_total // (3 * len(RISKY_QUERIES))):
            item = rng.choice(items)
            query = item["query"]

            # Decide if this instance hallucinates
            hallucinates = rng.random() < 0.7  # 70% hallucination rate for risky queries
            if hallucinates:
                answer = item["halluc_answer"]
            else:
                answer = item["true_answer"]

            # Self-consistency: hallucinated answers are inconsistent
            n_samples = 5
            answers = []
            for s in range(n_samples):
                if hallucinates:
                    # Generate varied hallucinated answers
                    if rng.random() < 0.6:
                        answers.append(item["halluc_answer"])
                    else:
                        # Different hallucination
                        modified = re.sub(r'\d+', lambda m: str(int(m.group()) + rng.randint(-20, 20)), item["halluc_answer"])
                        answers.append(modified)
                else:
                    answers.append(item["true_answer"])

            consistency = 1.0 - np_rng.uniform(0.3, 0.7) if hallucinates else 1.0 - np_rng.uniform(0.0, 0.15)

            query_rows.append({
                "prompt_id": pid,
                "query": query,
                "response": answer,
                "domain": item["domain"],
                "is_hallucination": hallucinates,
                "hallucination_type": item["risk_type"] if hallucinates else "none",
                "risk_level": "high",
                "consistency_score": round(consistency, 3),
                "has_numeric_claim": bool(re.search(r'\d+', answer)),
                "has_named_entity": bool(re.search(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', answer)),
                "asks_about_person": any(w in query.lower() for w in ["who", "professor", "dr.", "artist"]),
                "asks_for_list": any(w in query.lower() for w in ["list", "name", "enumerate"]),
                "asks_specific_count": bool(re.search(r'how many|exactly \d+|list \d+', query.lower())),
                "query_length": len(query.split()),
                "response_length": len(answer.split()),
            })

            for s, ans in enumerate(answers):
                consistency_rows.append({
                    "prompt_id": pid, "sample_idx": s, "response": ans,
                })
            pid += 1

    # ── 2. Build RAG DataFrame ──
    rag_rows = []
    for scenario in RAG_SCENARIOS:
        # Generate multiple instances with varying faithfulness
        for _ in range(8):
            is_faithful = rng.random() < 0.5
            rag_rows.append({
                "query": scenario["query"],
                "context": scenario["context"],
                "response": scenario["faithful_response"] if is_faithful else scenario["unfaithful_response"],
                "is_faithful": is_faithful,
                "domain": scenario["domain"],
            })

    # ── 3. Build Drift DataFrame ──
    drift_rows = []
    for item in DRIFT_RESPONSES:
        for sent_idx, (sentence, trust, is_halluc) in enumerate(item["sentences"]):
            drift_rows.append({
                "query": item["query"],
                "sentence_idx": sent_idx,
                "sentence": sentence,
                "trust_score": trust,
                "is_hallucination": is_halluc,
                "drift_point": item["drift_point"],
            })

    queries_df = pd.DataFrame(query_rows)
    consistency_df = pd.DataFrame(consistency_rows)
    rag_df = pd.DataFrame(rag_rows)
    drift_df = pd.DataFrame(drift_rows)

    n_halluc = queries_df["is_hallucination"].sum()
    print(f"Generated synthetic hallucination data:")
    print(f"  Queries: {len(queries_df)} ({n_halluc} hallucinated, {len(queries_df)-n_halluc} reliable)")
    print(f"  Consistency samples: {len(consistency_df)} ({len(consistency_df)//5} queries × 5)")
    print(f"  RAG scenarios: {len(rag_df)}")
    print(f"  Drift examples: {len(drift_df)} sentences")

    return {
        "queries": queries_df,
        "consistency": consistency_df,
        "rag": rag_df,
        "drift": drift_df,
    }
