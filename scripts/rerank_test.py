from llama_index import ServiceContext, PromptHelper
from llama_index.node_parser import SimpleNodeParser
from llama_index.postprocessor import SentenceTransformerRerank

from qasper_data.qasper_entity import PARAGRAPH_SEPARATOR
from qasper_data.qasper_dataset import QasperDataset, PaperIndex

from qasper_data.qasper_evaluator import EvidenceEvaluator, HitEvaluator, AnswerEvaluator

from tqdm import tqdm

import pandas as pd

prompt_helper = PromptHelper(context_window=4096, num_output=3)

node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=0,
                                             paragraph_separator=PARAGRAPH_SEPARATOR)

context = ServiceContext.from_defaults(llm="local", embed_model="local",
                                       prompt_helper=prompt_helper,
                                       node_parser=node_parser)

# evidence_evaluator = EvidenceEvaluator.from_defaults(context)
# hit_evaluator = HitEvaluator.from_defaults(context)
answer_evaluator = AnswerEvaluator.from_defaults(context)

papers = QasperDataset("test").as_papers()

df_scores = pd.DataFrame(columns=["question", "answer-f1", "context-recall", "context-precision"])

for paper in tqdm(papers[:100]):
    paper_index = PaperIndex(paper, context)
    query_engine = paper_index.as_index().as_query_engine(mode="no_text",
                                                          similarity_top_k=8,
                                                          node_postprocessors=[
                                                              SentenceTransformerRerank(
                                                                  model="cross-encoder/ms-marco-MiniLM-L-12-v2",
                                                                  top_n=3,
                                                              ),
                                                          ])
    dataframe = answer_evaluator.evaluate(paper, query_engine)
    # dataframe = evidence_evaluator.evaluate(paper, query_engine)
    # hit_df = hit_evaluator.evaluate(paper, query_engine)
    # join the two dataframes on the question column
    # dataframe = pd.merge(evidence_df, hit_df, on="question")
    if df_scores.empty:
        df_scores = dataframe
    else:
        df_scores = pd.concat([df_scores, dataframe])
    del paper_index

df_scores.to_csv("baseline_rerank_answer.csv", index=False)
