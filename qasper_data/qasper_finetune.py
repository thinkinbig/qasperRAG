from typing import Any, Optional, List

import numpy as np
from llama_index import ServiceContext
from llama_index.embeddings import BaseEmbedding, resolve_embed_model
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from llama_index.finetuning.cross_encoders.dataset_gen import CrossEncoderFinetuningDatasetSample
from llama_index.finetuning.types import BaseEmbeddingFinetuneEngine
from llama_index.schema import MetadataMode
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from qasper_data.qasper_entity import Paper, PARAGRAPH_SEPARATOR
from qasper_data.qasper_dataset import PaperIndex
from qasper_data.qasper_evaluator import normalize_string
from llama_index.node_parser import NodeParser, SimpleNodeParser


def generate_qa_embedding_pairs(context: ServiceContext, paper: Paper) -> EmbeddingQAFinetuneDataset:
    """
    Generate examples given a set of nodes.
    """
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=512,
        chunk_overlap=0,
        paragraph_separator=PARAGRAPH_SEPARATOR)
    nq_relevancy = NQRelevancy(paper, node_parser, context)
    nodes_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nq_relevancy.nodes
    }
    queries = {
        question_id: question
        for question_id, question in zip(paper.qas.question_id, paper.qas.question)
    }
    relevant_docs = {
        question_id: [
            node.node_id for node, score in zip(nq_relevancy.nodes, nq_relevancy[i])
            if score == 1
        ]
        for i, question_id in enumerate(paper.qas.question_id)
    }
    return EmbeddingQAFinetuneDataset(
        queries=queries,
        corpus=nodes_dict,
        relevant_docs=relevant_docs,
    )


def generate_qa_cross_encoder_pairs(context: ServiceContext, paper: Paper) -> List[CrossEncoderFinetuningDatasetSample]:
    """
    Generate examples given a set of nodes.
    """
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=512,
        chunk_overlap=0,
        paragraph_separator=PARAGRAPH_SEPARATOR)
    nq_relevancy = NQRelevancy(paper, node_parser, context)
    return [CrossEncoderFinetuningDatasetSample(
        query=nq_relevancy.question_text(j),
        context=nq_relevancy.node_text(i),
        score=nq_relevancy[i][j])
        for j in range(nq_relevancy.N) for i in range(nq_relevancy.M) if not np.all(nq_relevancy[i] == 0)]


class NQRelevancy:
    """
    Class for computing the relevancy Table of Nodes and Questions
    """

    def __init__(self,
                 paper: Paper,
                 node_parser: NodeParser,
                 service_context: ServiceContext):
        self.paper = paper
        self.nodes = node_parser.get_nodes_from_documents(PaperIndex(paper, service_context).as_documents())
        self.question_ids = paper.qas.question_id
        self.M = len(self.nodes)
        self.N = len(self.question_ids)
        self.n_q_matrix = np.zeros((self.M, self.N))
        self.compute_relevancy()

    def node_text(self, index: int):
        return self.nodes[index].get_content(MetadataMode.ALL)

    def node_id(self, index: int):
        return self.nodes[index].node_id

    def question_text(self, index: int):
        return self.paper.qas.question[index]

    def question_id(self, index: int):
        return self.paper.qas.question_id[index]

    def compute_relevancy(self):
        for question_id in self.question_ids:
            evidences = self.paper.get_evidences_by_question_id(question_id)
            for evidence in evidences:
                for node in self.nodes:
                    if normalize_string(evidence) in normalize_string(node.get_content(MetadataMode.ALL)):
                        self.n_q_matrix[self.nodes.index(node)][self.question_ids.index(question_id)] = 1

    def __getitem__(self, item):
        return self.n_q_matrix[item]

    def __call__(self, *args, **kwargs):
        return self.n_q_matrix


class MyEmbeddingFinetuneEngine(BaseEmbeddingFinetuneEngine):

    def __init__(self,
                 dataset: EmbeddingQAFinetuneDataset,
                 model_id: str = "BAAI/bge-small-en",
                 model_output_path: str = "exp_finetune",
                 batch_size: int = 10,
                 val_dataset: Optional[EmbeddingQAFinetuneDataset] = None,
                 loss: Optional[Any] = None,
                 epochs: int = 2,
                 show_progress_bar: bool = True,
                 evaluation_steps: int = 50,
                 ):
        self.dataset = dataset

        self.model_id = model_id
        self.model_output_path = model_output_path
        self.model = SentenceTransformer(model_id)

        examples: Any = []
        for query_id, query in dataset.queries.items():
            for node_id in dataset.relevant_docs[query_id]:
                text = dataset.corpus[node_id]
                example = InputExample(texts=[query, text])
                examples.append(example)
        self.examples = examples

        self.loader: DataLoader = DataLoader(examples, batch_size=batch_size)

        # define evaluator
        from sentence_transformers.evaluation import InformationRetrievalEvaluator

        evaluator: Optional[InformationRetrievalEvaluator] = None
        if val_dataset is not None:
            evaluator = InformationRetrievalEvaluator(
                val_dataset.queries, val_dataset.corpus, val_dataset.relevant_docs
            )
        self.evaluator = evaluator

        # define loss
        self.loss = loss or losses.MultipleNegativesRankingLoss(self.model)

        self.epochs = epochs
        self.show_progress_bar = show_progress_bar
        self.evaluation_steps = evaluation_steps
        self.warmup_steps = int(len(self.loader) * epochs * 0.1)

    def finetune(self, **train_kwargs: Any) -> None:
        """Finetune model."""
        self.model.fit(
            train_objectives=[(self.loader, self.loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            output_path=self.model_output_path,
            show_progress_bar=self.show_progress_bar,
            evaluator=self.evaluator,
            evaluation_steps=self.evaluation_steps,
        )

    def get_finetuned_model(self, **model_kwargs: Any) -> BaseEmbedding:
        """Gets finetuned model."""
        embed_model_str = "local:" + self.model_output_path
        return resolve_embed_model(embed_model_str)
