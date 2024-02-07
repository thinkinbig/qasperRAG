from llama_index import ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.retrievers import BM25Retriever

from qasper_data.qasper_entity import PARAGRAPH_SEPARATOR
from qasper_data.qasper_dataset import QasperDataset, PaperIndex

from qasper_data.qasper_evaluator import EvidenceEvaluator, HitEvaluator, AnswerEvaluator
from qasper_data.qasper_prompt import DEFAULT_LLAMA_TEXT_QA_PROMPT, DEFAULT_LLAMA_REFINE_PROMPT

from tqdm import tqdm

import pandas as pd

node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=0, paragraph_separator=PARAGRAPH_SEPARATOR)

context = ServiceContext.from_defaults(llm="local", embed_model="local", node_parser=node_parser)

evidence_evaluator = EvidenceEvaluator.from_defaults(context)
answer_evaluator = AnswerEvaluator.from_defaults(context)
hit_evaluator = HitEvaluator.from_defaults(context)

papers = QasperDataset("test").random_sample(200)

# df_scores = pd.DataFrame(columns=["question", "evidence-f1", "context-recall", "context-precision"])
df_scores = pd.DataFrame(columns=["question", "answer-f1", "answer-recall", "answer-precision"])

for paper in tqdm(papers):
    paper_index = PaperIndex(paper, context)
    retriever = BM25Retriever.from_defaults(paper_index.as_index(), similarity_top_k=3)
    query_engine = paper_index.as_index().as_query_engine(
        similarity_top_k=3,
        retriever=retriever,
        text_qa_template=DEFAULT_LLAMA_TEXT_QA_PROMPT,
        refine_template=DEFAULT_LLAMA_REFINE_PROMPT,
    )
    # dataframe = evidence_evaluator.evaluate(paper, query_engine)
    # hit_df = hit_evaluator.evaluate(paper, query_engine)
    # join the two dataframes on the question column
    # dataframe = pd.merge(evidence_df, hit_df, on="question")
    dataframe = answer_evaluator.evaluate(paper, query_engine)
    if df_scores.empty:
        df_scores = dataframe
    else:
        df_scores = pd.concat([df_scores, dataframe])
    del paper_index

# df_scores.to_csv("bm25_retriever.csv", index=False)
df_scores.to_csv("bm25_retriever_answer.csv", index=False)
