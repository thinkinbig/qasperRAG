from enum import Enum
from typing import List

from pydantic import BaseModel

PARAGRAPH_SEPARATOR = "\n\n"


class Section(BaseModel):
    section_name: List[str]
    paragraphs: List[List[str]]

    def __init__(self, **data):
        # Remove None values in section_name and paragraphs
        data["section_name"] = [name for name in data["section_name"] if name is not None]
        data["paragraphs"] = [paragraph for paragraph in data["paragraphs"] if paragraph is not None]
        super().__init__(**data)


class Answer(BaseModel):
    unanswerable: bool
    extractive_spans: List[str]
    yes_no: bool | None
    free_form_answer: str
    evidence: List[str]
    highlighted_evidence: List[str]

    @property
    def answer_type(self):
        if self.unanswerable:
            return AnswerType.NONE
        elif self.yes_no is not None:
            return AnswerType.BOOLEAN
        elif self.extractive_spans:
            return AnswerType.EXTRACTIVE
        else:
            return AnswerType.ABSTRACTIVE

    @property
    def answer_string(self):
        if self.answer_type == AnswerType.EXTRACTIVE:
            return ", ".join(self.extractive_spans)
        elif self.answer_type == AnswerType.ABSTRACTIVE:
            return self.free_form_answer
        elif self.answer_type == AnswerType.BOOLEAN:
            return "Yes" if self.yes_no else "No"
        else:
            return "Unacceptable"


class Answers(BaseModel):
    answer: List[Answer]
    annotation_id: List[str]
    worker_id: List[str]


class QA(BaseModel):
    question: List[str]
    question_id: List[str]
    nlp_background: List[str]
    topic_background: List[str]
    paper_read: List[str]
    search_query: List[str]
    question_writer: List[str]
    answers: List[Answers]


class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    full_text: Section
    qas: QA

    def get_formatted_full_text(self) -> str:
        """
        Combine all paragraphs and section name in the paper into a single string in the format:
        ======Section Name=======
        Paragraph 1
        Paragraph 2
        ...

        :return: the full text of the paper
        """
        full_text = ""
        full_text += f"\n======={self.title}=======\n" + "".join(self.abstract)
        for section_name, paragraphs in zip(self.full_text.section_name, self.full_text.paragraphs):
            full_text += f"\n======={section_name}=======\n" + "".join(paragraphs)
        return full_text

    def get_full_text(self) -> str:
        """
        Combine all paragraphs in the paper into a single string.
        :return: the full text of the paper
        """
        full_text = ""
        full_text += "".join(self.abstract) + PARAGRAPH_SEPARATOR
        for paragraphs in self.full_text.paragraphs:
            full_text += "".join(paragraphs) + PARAGRAPH_SEPARATOR
        return full_text

    def get_evidences_by_question_id(self, question_id: str) -> List[str]:
        """
        Get the list of evidences for a question in the paper.
        :param question_id: the question id
        :return: the list of evidences
        """
        evidences = []
        qas = self.qas
        assert len(qas.question) == len(qas.question_id) == len(qas.answers)
        for i, qid in enumerate(qas.question_id):
            if qid == question_id:
                for j in range(len(qas.answers[i].answer)):
                    for evidence in qas.answers[i].answer[j].evidence:
                        evidences.append(evidence)
        return evidences

    def get_evidences_by_question(self, question: str) -> List[str]:
        """
        Get the list of evidences for a question in the paper.
        :param question: the question
        :return: the list of evidences
        """
        evidences = []
        qas = self.qas
        assert len(qas.question) == len(qas.question_id) == len(qas.answers)
        for i, q in enumerate(qas.question):
            if q == question:
                for j in range(len(qas.answers[i].answer)):
                    for evidence in qas.answers[i].answer[j].evidence:
                        evidences.append(evidence)
        return evidences


class AnswerType(str, Enum):
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    BOOLEAN = "boolean"
    NONE = "none"
