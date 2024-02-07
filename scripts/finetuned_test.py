from llama_index import ServiceContext
from llama_index.embeddings import resolve_embed_model
from llama_index.node_parser import SimpleNodeParser

from qasper_data.qasper_entity import PARAGRAPH_SEPARATOR
from qasper_data.qasper_dataset import QasperDataset, PaperIndex

from qasper_data.qasper_evaluator import EvidenceEvaluator, AnswerEvaluator

from tqdm import tqdm

import pandas as pd


node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=0, paragraph_separator=PARAGRAPH_SEPARATOR)

# context = ServiceContext.from_defaults(llm="local", embed_model="local", node_parser=node_parser)

query_context = ServiceContext.from_defaults(llm="local",
                                             embed_model=resolve_embed_model("local:../models/fine_tuned_embedding"),
                                             node_parser=node_parser)

# evidence_evaluator = EvidenceEvaluator.from_defaults(context)
answer_evaluator = AnswerEvaluator.from_defaults(query_context)

papers = QasperDataset("test").as_papers()

# df_scores = pd.DataFrame(columns=["question", "evidence-f1", "context-recall", "context-precision", "hit"])

df_scores = pd.DataFrame(columns=["question", "answer-f1", "answer-recall", "answer-precision"])

for paper in tqdm(papers):
    paper_index = PaperIndex(paper, query_context)
    query_engine = paper_index.as_index().as_query_engine()
    # dataframe = evidence_evaluator.evaluate(paper, query_engine)
    dataframe = answer_evaluator.evaluate(paper, query_engine)
    # hit_df = hit_evaluator.evaluate(paper, query_engine)
    # join the two dataframes on the question column
    # dataframe = pd.merge(evidence_df, hit_df, on="question")
    if df_scores.empty:
        df_scores = dataframe
    else:
        df_scores = pd.concat([df_scores, dataframe])
    del paper_index

df_scores.to_csv("finetuned_emb_answer.csv", index=False)