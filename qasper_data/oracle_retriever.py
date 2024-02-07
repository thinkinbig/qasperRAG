from typing import List, Optional

from llama_index import QueryBundle, ServiceContext
from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core import BaseRetriever
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import NodeWithScore, BaseNode

from qasper_data.qasper_dataset import PaperIndex
from qasper_data.qasper_entity import Paper, PARAGRAPH_SEPARATOR


class OracleRetriever(BaseRetriever):

    def __init__(self,
                 nodes: List[BaseNode],
                 paper: Paper,
                 similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K) -> None:
        self._paper = paper
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k

    @classmethod
    def from_defaults(cls,
                      paper: Paper,
                      service_context: Optional[ServiceContext] = None
                      ) -> "OracleRetriever":
        service_context = service_context or ServiceContext.from_defaults(llm="local", embed_model="local")
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=0,
                                                     paragraph_separator=PARAGRAPH_SEPARATOR)
        paper_index = PaperIndex(paper, service_context)
        nodes = node_parser.get_nodes_from_documents(paper_index.as_documents())
        return cls(nodes=nodes, paper=paper)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        question = query_bundle.query_str
        evidences = self._paper.get_evidences_by_question(question)
        list_of_nodes = []
        for node in self._nodes:
            for evidence in evidences:
                if evidence.strip() in node.get_content().strip():
                    list_of_nodes.append(node)
            list_of_nodes = list_of_nodes[:self._similarity_top_k]
        return [NodeWithScore(node=node, score=1) for node in list_of_nodes]
