"""
LLM-as-a-Judge — GPT-4 based answer quality evaluation.

Following the standard methodology used in modern RAG papers:
the judge LLM evaluates generated answers on Correctness,
Completeness, Hallucination, and Specificity (1-5 scale).
"""

import logging
from typing import Optional, List
from dataclasses import dataclass

from llm_interface.llm_client import LLMClient
from query_engine.prompt_templates import JUDGE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    """Evaluation scores from the LLM judge."""
    correctness: float      # 1-5: factual accuracy
    completeness: float     # 1-5: covers all aspects
    hallucination: float    # 1-5: absence of fabrication (5 = none)
    specificity: float      # 1-5: references specific code elements
    reasoning: str = ""     # Judge's explanation
    avg_score: float = 0.0  # Average of all criteria

    def __post_init__(self):
        self.avg_score = (
            self.correctness + self.completeness +
            self.hallucination + self.specificity
        ) / 4.0


class LLMJudge:
    """
    Uses an LLM (GPT-4) to evaluate answer quality.

    Standard evaluation procedure in RAG research:
    1. Provide question, ground truth, and generated answer
    2. LLM rates on 4 criteria (1-5 each)
    3. Aggregate scores across all queries
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client or LLMClient()

    def evaluate(
        self,
        question: str,
        generated_answer: str,
        ground_truth: str,
    ) -> JudgeScore:
        """
        Evaluate a single answer using the LLM judge.

        Returns JudgeScore with ratings on all 4 criteria.
        """
        prompt = JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            generated_answer=generated_answer,
        )

        response = self._llm.query(prompt, temperature=0.0, max_tokens=500)
        parsed = self._llm.parse_json_response(response)

        if parsed:
            return JudgeScore(
                correctness=float(parsed.get("correctness", 3)),
                completeness=float(parsed.get("completeness", 3)),
                hallucination=float(parsed.get("hallucination", 3)),
                specificity=float(parsed.get("specificity", 3)),
                reasoning=parsed.get("reasoning", ""),
            )
        else:
            logger.warning("Judge response could not be parsed, using defaults")
            return JudgeScore(
                correctness=3.0,
                completeness=3.0,
                hallucination=3.0,
                specificity=3.0,
                reasoning="Parse error — defaulting to neutral scores",
            )

    def evaluate_batch(
        self,
        questions: List[str],
        generated_answers: List[str],
        ground_truths: List[str],
    ) -> List[JudgeScore]:
        """Evaluate a batch of answers."""
        scores = []
        for q, gen, gt in zip(questions, generated_answers, ground_truths):
            score = self.evaluate(q, gen, gt)
            scores.append(score)
        return scores

    @staticmethod
    def aggregate_scores(scores: List[JudgeScore]) -> dict:
        """Compute aggregate statistics over judge scores."""
        if not scores:
            return {}

        n = len(scores)
        return {
            "avg_correctness": sum(s.correctness for s in scores) / n,
            "avg_completeness": sum(s.completeness for s in scores) / n,
            "avg_hallucination": sum(s.hallucination for s in scores) / n,
            "avg_specificity": sum(s.specificity for s in scores) / n,
            "avg_overall": sum(s.avg_score for s in scores) / n,
            "num_evaluated": n,
        }
