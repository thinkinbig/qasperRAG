import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

from llama_index.finetuning import EmbeddingQAFinetuneDataset


@dataclass
class Section:
    section_name: str
    paragraphs: List[str]

    def __getitem__(self, item):
        if item == "section_name":
            return self.section_name
        elif item == "paragraphs":
            return self.paragraphs
        else:
            raise KeyError("Invalid key")

    def to_json(self):
        return {
            "section_name": self.section_name,
            "paragraphs": self.paragraphs
        }

    @staticmethod
    def from_json(json_obj):
        return Section(section_name=json_obj["section_name"], paragraphs=json_obj["paragraphs"])


@dataclass
class Answer:
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

    @property
    def local_answer(self) -> str:
        """
        The Answers field in the dataset follow the below format:-
        Unanswerable answers have "unanswerable" set to true.

        The remaining answers have exactly one of the following fields being non-empty.

        "extractive_spans" are spans in the paper which serve as the answer.
        "free_form_answer" is a written out answer.
        "yes_no" is true iff the answer is Yes, and false iff the answer is No.

        We accept only free-form answers and for all the other kind of answers we set their value to 'Unacceptable'
        :return: the answer string
        """
        if self.answer_type == AnswerType.ABSTRACTIVE:
            return self.free_form_answer
        else:
            return "Unacceptable"

    def __getitem__(self, item):
        if item == "answer_type":
            return self.answer_type
        elif item == "answer_string":
            return self.answer_string
        elif item == "local_answer":
            return self.local_answer
        elif item == "unanswerable":
            return self.unanswerable
        elif item == "extractive_spans":
            return self.extractive_spans
        elif item == "yes_no":
            return self.yes_no
        elif item == "free_form_answer":
            return self.free_form_answer
        elif item == "evidence":
            return self.evidence
        elif item == "highlighted_evidence":
            return self.highlighted_evidence
        else:
            raise KeyError("Invalid key")

    def to_json(self):
        return {
            "unanswerable": self.unanswerable,
            "extractive_spans": self.extractive_spans,
            "yes_no": self.yes_no,
            "free_form_answer": self.free_form_answer,
            "evidence": self.evidence,
            "highlighted_evidence": self.highlighted_evidence,
            "answer_type": self.answer_type.value,
            "answer_string": self.answer_string
        }

    @staticmethod
    def from_json(json_obj):
        return Answer(unanswerable=json_obj["unanswerable"], extractive_spans=json_obj["extractive_spans"],
                      yes_no=json_obj["yes_no"], free_form_answer=json_obj["free_form_answer"],
                      evidence=json_obj["evidence"], highlighted_evidence=json_obj["highlighted_evidence"])


@dataclass
class Question:
    question: str
    question_id: str
    nlp_background: str
    topic_background: str
    paper_read: str
    search_query: str
    question_writer: str
    answers: List[Answer]

    def __getitem__(self, item):
        if item == "question":
            return self.question
        elif item == "question_id":
            return self.question_id
        elif item == "answers":
            return self.answers
        else:
            raise KeyError("Invalid key")

    def to_json(self):
        return {
            "question": self.question,
            "question_id": self.question_id,
            "nlp_background": self.nlp_background,
            "topic_background": self.topic_background,
            "paper_read": self.paper_read,
            "search_query": self.search_query,
            "question_writer": self.question_writer,
            "answers": [answer.to_json() for answer in self.answers]
        }

    @staticmethod
    def from_json(json_obj):
        return Question(question=json_obj["question"], question_id=json_obj["question_id"],
                        nlp_background=json_obj["nlp_background"], topic_background=json_obj["topic_background"],
                        paper_read=json_obj["paper_read"], search_query=json_obj["search_query"],
                        question_writer=json_obj["question_writer"],
                        answers=[Answer.from_json(answer) for answer in json_obj["answers"]])


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    full_text: List[Section]
    qas: List[Question]

    def __getitem__(self, item):
        if item == "id":
            return self.id
        elif item == "title":
            return self.title
        elif item == "abstract":
            return self.abstract
        elif item == "full_text":
            return self.full_text
        elif item == "qas":
            return self.qas
        else:
            raise KeyError("Invalid key")

    def to_json(self):
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "full_text": [section.to_json() for section in self.full_text],
            "qas": [question.to_json() for question in self.qas]
        }

    @staticmethod
    def from_json(json_obj):
        return Paper(id=json_obj["id"], title=json_obj["title"], abstract=json_obj["abstract"],
                     full_text=[Section.from_json(section) for section in json_obj["full_text"]],
                     qas=[Question.from_json(question) for question in json_obj["qas"]])

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
        for section in self.full_text:
            full_text += f"\n======={section.section_name}=======\n" + "".join(section.paragraphs)
        return full_text

    def get_full_text(self) -> str:
        """
        Combine all paragraphs in the paper into a single string.
        :return: the full text of the paper
        """
        # return "\t".join([paragraph for section in self.full_text for paragraph in section.paragraphs])
        full_text = ""
        for section in self.full_text:
            full_text += "".join(section.paragraphs) + "\n"
        return full_text

    def get_questions(self) -> List[str]:
        """
        Get all questions in the paper.
        :return: a list of questions as string in the paper
        """
        return [qa.question for qa in self.qas]

    def get_answers(self) -> List[List[str]]:
        """
        Get all answers in the paper.
        :return: a list of answers as string in the paper
        """
        return [[answer.answer_string for answer in qa.answers] for qa in self.qas]


class MyEmbeddingQAFinetuneDataset(EmbeddingQAFinetuneDataset):
    relevant_answers: Dict[str, List[str]]

    @classmethod
    def from_json(cls, path: str) -> "MyEmbeddingQAFinetuneDataset":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            corpus=data["corpus"],
            queries=data["queries"],
            relevant_docs=data["relevant_docs"],
            relevant_answers=data["relevant_answers"],
        )

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=4)


class AnswerType(Enum):
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    BOOLEAN = "boolean"
    NONE = "none"


def ingest_dataset(dataset) -> List[Paper]:
    papers = []
    for paper in dataset:
        full_text = paper['full_text']
        paper_full_text = []
        assert len(full_text['section_name']) == len(full_text['paragraphs'])
        for i in range(len(full_text['section_name'])):
            section_name = full_text['section_name'][i]
            paragraphs = full_text['paragraphs'][i]
            paper_full_text.append(Section(section_name=section_name, paragraphs=paragraphs))
        qas = paper['qas']
        paper_qas = []
        for i in range(len(qas['question'])):
            question = qas['question'][i]
            question_id = qas['question_id'][i]
            nlp_background = qas['nlp_background'][i]
            topic_background = qas['topic_background'][i]
            paper_read = qas['paper_read'][i]
            search_query = qas['search_query'][i]
            question_writer = qas['question_writer'][i]
            answers = []
            for j in range(len(qas['answers'][i]['answer'])):
                unanswerable = qas['answers'][i]['answer'][j]['unanswerable']
                extractive_spans = qas['answers'][i]['answer'][j]['extractive_spans']
                yes_no = qas['answers'][i]['answer'][j]['yes_no']
                free_form_answer = qas['answers'][i]['answer'][j]['free_form_answer']
                evidence = qas['answers'][i]['answer'][j]['evidence']
                highlighted_evidence = qas['answers'][i]['answer'][j]['highlighted_evidence']
                answers.append(Answer(unanswerable=unanswerable, extractive_spans=extractive_spans, yes_no=yes_no,
                                      free_form_answer=free_form_answer, evidence=evidence,
                                      highlighted_evidence=highlighted_evidence))
            paper_qas.append(Question(question=question, question_id=question_id, nlp_background=nlp_background,
                                      topic_background=topic_background, paper_read=paper_read,
                                      search_query=search_query,
                                      question_writer=question_writer, answers=answers))
        papers.append(Paper(id=paper['id'], title=paper['title'], abstract=paper['abstract'], full_text=paper_full_text,
                            qas=paper_qas))
    return papers


class Prediction(Dict[str, Answer]):
    """
    A predicted answer for a single question.
    """

    def to_json(self):
        json_obj = {}
        for question_id, answer in self.items():
            json_obj[question_id] = answer.to_json()
        return json_obj

    @staticmethod
    def from_json(json_obj):
        prediction = Prediction()
        for question_id, answer in json_obj.items():
            prediction[question_id] = Answer.from_json(answer)
        return prediction


class Gold(Dict[str, List[Answer]]):
    """
    A list of gold answers for a single question.
    """

    def to_json(self):
        json_obj = {}
        for question_id, answers in self.items():
            json_obj[question_id] = [answer.to_json() for answer in answers]
        return json_obj

    @staticmethod
    def from_json(json_obj):
        gold = Gold()
        for question_id, answers in json_obj.items():
            gold[question_id] = [Answer.from_json(answer) for answer in answers]
        return gold


class Golds(Dict[str, Gold]):
    """
    A repository of gold answers, keyed by paper ID.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def from_dataset(dataset):
        from copy import copy
        golds = Golds()

        for paper in dataset:
            gold = Gold()
            for qa in paper.qas:
                gold[qa.question_id] = [copy(answer) for answer in qa.answers]
            golds[paper.id] = gold
        return golds

    def to_json(self):
        json_obj = {}
        for paper_id, gold in self.items():
            json_obj[paper_id] = gold.to_json()
        return json_obj

    @staticmethod
    def from_json(json_obj):
        golds = Golds()
        for paper_id, gold in json_obj.items():
            golds[paper_id] = Gold.from_json(gold)
        return golds

    def merge(self) -> Gold:
        """
        Merge all the golds into one gold for the entire dataset
        :return: Gold
        """
        gold = Gold()
        for paper_id, paper_gold in self.items():
            for question_id, question_gold in paper_gold.items():
                assert question_id not in gold
                gold[question_id] = question_gold
        return gold


class Predictions(Dict[str, Prediction]):
    """
    A repository of predicted answers, keyed by paper ID.
    """

    def __init__(self):
        super().__init__()

    def to_json(self):
        json_obj = {}
        for paper_id, prediction in self.items():
            json_obj[paper_id] = prediction.to_json()
        return json_obj

    @staticmethod
    def from_json(json_obj):
        predictions = Predictions()
        for paper_id, prediction in json_obj.items():
            predictions[paper_id] = Prediction.from_json(prediction)
        return predictions

    def merge(self) -> Prediction:
        """
        Merge all the predictions into one prediction for the entire dataset
        :return: Prediction
        """
        prediction = Prediction()
        for paper_id, paper_prediction in self.items():
            for question_id, question_prediction in paper_prediction.items():
                assert question_id not in prediction
                prediction[question_id] = question_prediction
        return prediction


class Query:
    """
    A question query.
    """
    query: str

    def __init__(self, query: str):
        self.query = query

    def to_json(self):
        return {
            "query": self.query,
        }

    @staticmethod
    def from_json(json_obj):
        return Query(json_obj["query"])


class Queries(Dict[str, Query]):
    """
    A repository of questions, keyed by query id.
    """

    def __init__(self):
        super().__init__()

    def to_json(self):
        json_obj = {}
        for query_id, query in self.items():
            json_obj[query_id] = query.to_json()
        return json_obj

    @staticmethod
    def from_json(json_obj):
        questions = Queries()
        for query_id, query in json_obj.items():
            questions[query_id] = Query.from_json(query)
        return questions

    @staticmethod
    def from_paper(paper: Paper):
        queries = Queries()
        for qa in paper.qas:
            queries[qa.question_id] = Query(qa.question)
        return queries
