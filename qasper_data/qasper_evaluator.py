import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional, List

import pandas as pd
from llama_index import ServiceContext
from llama_index.query_engine import RetrieverQueryEngine

from qasper_data.data import Paper, AnswerType


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def mean(x):
    return sum(x) / len(x) if x else 0.0


def paragraph_f1_score(prediction: str, ground_truth: str):
    if not ground_truth and not prediction:
        # The question is unanswerable and the prediction is empty.
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class BaseQasperEvaluator(ABC):
    """
    Qasper Evaluator designed for Paper object
    """

    def __init__(self, service_context: Optional[ServiceContext] = None):
        self.service_context = (service_context or
                                ServiceContext.from_defaults(embed_model="sentence-transformers/all-MiniLM-L6-v2"))

    @classmethod
    def from_defaults(cls, service_context: Optional[ServiceContext] = None):
        return cls(service_context=service_context)

    @abstractmethod
    def evaluate(self, paper: Paper, query_engine: RetrieverQueryEngine) -> List[float]:
        """
        Evaluate the paper and return the score for each question
        :param query_engine: Query engine to use
        :param paper: Paper object
        :return: List of scores for each question
        """
        pass


class EvidenceQasperEvaluator(BaseQasperEvaluator):
    """
    Calculate the F1 score of the evidence sentences
    By default only the answerable questions are considered
    """

    def evaluate(self, paper: Paper, query_engine: RetrieverQueryEngine) -> List[float]:
        """
        Calculate the F1 score of the evidence sentences
        By default only the answerable questions are considered
        """
        scores = []
        for qa in paper.qas:
            query = qa.question
            nodes = query_engine.query(query).source_nodes
            texts = [node.text for node in nodes]
            score: float = 0.0
            for answer in qa.answers:
                evidences = answer.evidence
                # concatenate all the evidence sentences
                for reference in evidences:
                    for text in texts:
                        f1 = paragraph_f1_score(text, reference)
                        score = max(score, f1)
            # append the highest score for this question
            scores.append(score)
        return scores

    @staticmethod
    def get_evaluation_dataframe(paper: Paper, query_engine: RetrieverQueryEngine) -> pd.DataFrame:
        """
        Calculate the F1 score of the evidence sentences
        By default only the answerable questions are considered
        """
        df_scores = pd.DataFrame(columns=["question", "context", "score"])
        for qa in paper.qas:
            question = qa.question
            nodes = query_engine.query(question).source_nodes
            for node in nodes:
                text = node.text
                for answer in qa.answers:
                    if answer.answer_type == AnswerType.NONE or answer.answer_type == AnswerType.BOOLEAN:
                        continue
                    evidences = answer.evidence
                    score = 0.0
                    for reference in evidences:
                        normalized_reference = normalize_answer(reference)
                        normalized_text = normalize_answer(text)

                        if normalized_reference in normalized_text:
                            score = 1.0
                            break
                    if df_scores.empty:
                        df_scores = pd.DataFrame([[question, text, score]],
                                                 columns=["question", "context", "score"])
                    else:
                        df_scores = pd.concat([df_scores, pd.DataFrame([[question, text, score]],
                                                                       columns=["question", "context", "score"])],
                                              ignore_index=True)
        return df_scores

