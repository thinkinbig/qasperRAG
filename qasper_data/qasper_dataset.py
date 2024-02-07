import random
from enum import Enum
from typing import Optional, List

from datasets import load_dataset
from llama_index import VectorStoreIndex, ServiceContext, Document
from torch.utils.data import Dataset

from qasper_data.qasper_entity import Paper


class QasperType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


qasper_dataset = load_dataset("allenai/qasper")


class QasperDataset(Dataset):
    """
    Qasper dataset that ingest the qasper paper
    By default, it will load the training dataset and the sequence of the papers is the same as the original dataset
    """

    def __init__(self, data_type: Optional[str] = "train", seed: Optional[int] = None):
        if data_type not in [QasperType.TRAIN.value, QasperType.VALIDATION.value, QasperType.TEST.value]:
            raise ValueError(f"Unsupported data type: {data_type}")
        self.dataset = qasper_dataset[data_type]
        # get a random seed if none is provided, defaulting to the current microsecond
        self.rand_seed = seed or 42

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def as_papers(self) -> List[Paper]:
        """
        Convert the dataset into a list of papers
        :return: a list of papers
        """
        return [Paper(**paper) for paper in self.dataset]

    def random_sample(self, n: int) -> List[Paper]:
        """
        Sample n random papers from the dataset
        :param n: number of papers to sample
        :return: a list of sampled papers
        """
        random.seed(self.rand_seed)
        if n > len(self.dataset):
            raise ValueError(f"Sample size {n} is larger than the dataset size {len(self.dataset)}")
        return random.sample(self.as_papers(), n)

    def get_paper_by_id(self, paper_id: str) -> Paper:
        """
        Get a paper by its id
        :param paper_id: the id of the paper to retrieve
        :return: the paper with the given id
        """
        return next(paper for paper in self.as_papers() if paper.id == paper_id)

    def match_paper_by_title(self, title: str) -> Paper:
        """
        Get a paper by its title
        :param title: the title of the paper to retrieve
        :return: the paper with the given title
        """
        return next(paper for paper in self.as_papers() if paper.title.strip() == title.strip())


class PaperIndex:
    def __init__(self, paper: Paper, service_context: ServiceContext | None = None):
        self.index = None
        self.documents = None
        self.service_context = service_context or ServiceContext.from_defaults()
        self.paper = paper

    def as_documents(self) -> List[Document]:
        if self.documents is None:
            documents = [Document(text=self.paper.get_full_text())]
            self.documents = documents
        return self.documents

    def as_index(self, **kwargs):
        if self.index is None:
            self.index = VectorStoreIndex.from_documents(self.as_documents(),
                                                         service_context=self.service_context,
                                                         **kwargs)
        return self.index

    def as_vector_store(self):
        return self.as_index().vector_store
