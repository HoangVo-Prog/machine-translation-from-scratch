"""
ROUGE metrics via external library API (`rouge-score`).
"""

from __future__ import annotations

from typing import Dict, List

from rouge_score import rouge_scorer, scoring


def corpus_rouge_scores(hypotheses: List[str], references: List[List[str]]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    aggregator = scoring.BootstrapAggregator()

    for hypothesis, refs in zip(hypotheses, references):
        if not refs:
            continue
        reference = refs[0]
        aggregator.add_scores(scorer.score(reference, hypothesis))

    result = aggregator.aggregate()
    return {
        "rouge1": 100.0 * result["rouge1"].mid.fmeasure,
        "rouge2": 100.0 * result["rouge2"].mid.fmeasure,
        "rougeL": 100.0 * result["rougeL"].mid.fmeasure,
    }
