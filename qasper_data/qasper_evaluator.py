import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional, Dict, Tuple

import pandas as pd
from llama_index import ServiceContext, QueryBundle
from llama_index.query_engine import RetrieverQueryEngine

from qasper_data.qasper_entity import Paper, AnswerType


def normalize_string(s):
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


def token_f1_score(prediction, ground_truth) -> Tuple[float, float, float]:
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_string(prediction).split()
    ground_truth_tokens = normalize_string(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def mean(x):
    return sum(x) / len(x) if x else 0.0


def paragraph_f1_score(prediction: str, ground_truth: str):
    num_same = len(set(normalize_string(ground_truth)).intersection(set(normalize_string(prediction))))
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


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
    def evaluate(self, paper: Paper, query_engine: RetrieverQueryEngine):
        """
        Evaluate the paper and return the score for each question
        :param query_engine: Query engine to use, To accelerate the evaluation, the NO_TEXT mode is recommended
        :param paper: Paper object
        :return: scores for each question
        """
        pass


class EvidenceEvaluator(BaseQasperEvaluator):
    """
    Calculate the F1 score and precision/recall of the evidence sentences
    By default only the answerable questions are considered
    """

    def evaluate(self, paper: Paper, query_engine: RetrieverQueryEngine):
        """
        Evaluate the paper and return the score for each question,
        for each question, the highest F1 score and respective precision/recall of the evidence sentences are returned
        """
        df_scores = pd.DataFrame(columns=["question", "evidence-f1", "context-recall", "context-precision"])
        qas = paper.qas
        assert len(qas.question) == len(qas.answers)
        for query, answers in zip(qas.question, qas.answers):
            nodes = query_engine.retrieve(QueryBundle(query))
            if len(nodes) == 0:
                continue
            texts = [node.text for node in nodes]
            max_f1 = 0.0
            f1_recall = 0.0
            f1_precision = 0.0
            for answer in answers.answer:
                evidences = answer.evidence
                # concatenate all the evidence sentences
                for retrieved_text in texts:
                    for evidence in evidences:
                        evidence_text = evidence
                        f1, precision, recall = token_f1_score(retrieved_text, evidence_text)
                        if f1 > max_f1:
                            max_f1 = f1
                            f1_recall = recall
                            f1_precision = precision
            if df_scores.empty:
                df_scores = pd.DataFrame([[query, max_f1, f1_recall, f1_precision]],
                                         columns=["question", "evidence-f1", "context-recall", "context-precision"])
            else:
                df_scores = pd.concat([df_scores, pd.DataFrame([[query, max_f1, f1_recall, f1_precision]],
                                                               columns=["question", "evidence-f1", "context-recall",
                                                                        "context-precision"])])

        return df_scores


def evaluate_response(answer: str, response: str) -> Tuple[float, float, float]:
    """
    Evaluate the response and return the F1 score and precision/recall of the answer sentences
    :param answer: Ground truth answer
    :param response: Response from the query engine
    :return: F1 score and precision/recall of the answer sentences
    """
    pattern = re.compile(r"^Based on the provided context information,")
    prediction = pattern.sub("", response.strip())
    f1, precision, recall = token_f1_score(prediction, answer)
    return f1, precision, recall


class AnswerEvaluator(BaseQasperEvaluator):
    """
    Calculate the F1 score and precision/recall of the answer sentences
    By default only the answerable questions are considered
    """

    def evaluate(self, paper: Paper, query_engine: RetrieverQueryEngine):
        """
        Evaluate the paper and return the score for each question,
        for each question, the highest F1 score and respective precision/recall of the answer sentences are returned
        """
        df_scores = pd.DataFrame(columns=["question", "answer-f1", "answer-recall", "answer-precision"])
        qas = paper.qas
        assert len(qas.question) == len(qas.answers)
        for query, answers in zip(qas.question, qas.answers):
            max_f1 = 0.0
            f1_recall = 0.0
            f1_precision = 0.0
            for answer in answers.answer:
                answer_text = answer.answer_string
                answer_type = answer.answer_type
                response = query_engine.query(QueryBundle(query))
                # leverage the fact that the answer is always at the beginning of the response text
                pattern = re.compile(r"^Based on the provided context information,")
                prediction = pattern.sub("", response.response.strip())
                f1, precision, recall = token_f1_score(prediction, answer_text)
                if f1 > max_f1:
                    max_f1 = f1
                    f1_recall = recall
                    f1_precision = precision
            if df_scores.empty:
                df_scores = pd.DataFrame([[query, max_f1, f1_recall, f1_precision]],
                                         columns=["question", "answer-f1", "answer-recall", "answer-precision"])
            else:
                df_scores = pd.concat([df_scores, pd.DataFrame([[query, max_f1, f1_recall, f1_precision]],
                                                               columns=["question", "answer-f1", "answer-recall",
                                                                        "answer-precision"])])

        return df_scores


class HitEvaluator(BaseQasperEvaluator):

    def evaluate(self, paper: Paper, query_engine: RetrieverQueryEngine):
        df_hit = pd.DataFrame(columns=["question", "hit"])
        qa = paper.qas
        assert len(qa.question) == len(qa.answers)
        for query, answers in zip(qa.question, qa.answers):
            nodes = query_engine.retrieve(QueryBundle(query))
            if len(nodes) == 0:
                continue
            texts = [node.text for node in nodes]
            max_hit = 0
            for answer in answers.answer:
                hit = 0
                evidences = answer.evidence
                # concatenate all the evidence sentences
                for retrieved_text in texts:
                    for evidence in evidences:
                        retrieved_text = normalize_string(retrieved_text)
                        evidence = normalize_string(evidence)
                        if evidence in retrieved_text:
                            hit += 1
                if hit > max_hit:
                    max_hit = hit
            if df_hit.empty:
                df_hit = pd.DataFrame([[query, max_hit]],
                                      columns=["question", "hit"])
            else:
                df_hit = pd.concat([df_hit, pd.DataFrame([[query, max_hit]],
                                                         columns=["question", "hit"])])
        return df_hit
