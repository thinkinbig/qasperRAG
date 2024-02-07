import pandas as pd
from llama_index.node_parser import SimpleNodeParser
from tqdm import tqdm

from qasper_data.qasper_entity import PARAGRAPH_SEPARATOR

# chunk_512_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=0,
#                                                   paragraph_separator=PARAGRAPH_SEPARATOR)
chunk_1024_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=0,
                                                   paragraph_separator=PARAGRAPH_SEPARATOR)
# chunk_64_parser = SimpleNodeParser.from_defaults(chunk_size=64, chunk_overlap=0,
#                                                  paragraph_separator=PARAGRAPH_SEPARATOR)

from llama_index import ServiceContext

from qasper_data.qasper_dataset import QasperDataset, PaperIndex
from qasper_data.qasper_evaluator import EvidenceEvaluator, HitEvaluator, AnswerEvaluator

# context = ServiceContext.from_defaults(llm="local", embed_model="local", node_parser=chunk_512_parser)

# evidence_evaluator = EvidenceEvaluator.from_defaults(context)
# answer_evaluator = AnswerEvaluator.from_defaults(context)
# hit_evaluator = HitEvaluator.from_defaults(context)

papers = QasperDataset("test").as_papers()

# # scores_512 = pd.DataFrame(columns=["question", "evidence-f1", "context-recall", "context-precision"])
# scores_512 = pd.DataFrame(columns=["question", "answer-f1", "answer-recall", "answer-precision"])
#
# for paper in tqdm(papers[:10]):
#     paper_index = PaperIndex(paper, context)
#     query_engine = paper_index.as_index().as_query_engine()
#     # evidence_df = evidence_evaluator.evaluate(paper, query_engine)
#     # hit_df = hit_evaluator.evaluate(paper, query_engine)
#     # join the two dataframes on the question column
#     # dataframe = pd.merge(evidence_df, hit_df, on="question")
#     dataframe = answer_evaluator.evaluate(paper, query_engine)
#     if scores_512.empty:
#         scores_512 = dataframe
#     else:
#         scores_512 = pd.concat([scores_512, dataframe])
#     del paper_index

context = ServiceContext.from_defaults(llm="local", embed_model="local", node_parser=chunk_1024_parser)
evidence_evaluator = EvidenceEvaluator.from_defaults(context)
answer_evaluator = AnswerEvaluator.from_defaults(context)
hit_evaluator = HitEvaluator.from_defaults(context)
scores_1024 = pd.DataFrame(columns=["question", "evidence-f1", "context-recall", "context-precision"])

for paper in tqdm(papers[:10]):
    paper_index = PaperIndex(paper, context)
    query_engine = paper_index.as_index().as_query_engine()
    # scores = evidence_evaluator.evaluate(paper, query_engine)
    # scores_1024.extend(scores)
    # evidence_df = evidence_evaluator.evaluate(paper, query_engine)
    # hit_df = hit_evaluator.evaluate(paper, query_engine)
    # join the two dataframes on the question column
    # dataframe = pd.merge(evidence_df, hit_df, on="question")
    dataframe = answer_evaluator.evaluate(paper, query_engine)
    if scores_1024.empty:
        scores_1024 = dataframe
    else:
        scores_1024 = pd.concat([scores_1024, dataframe])
    del paper_index

# scores_64 = pd.DataFrame(columns=["question", "evidence-f1", "context-recall", "context-precision"])
#
# context = ServiceContext.from_defaults(llm="local", embed_model="local", node_parser=chunk_64_parser)
#
# evidence_evaluator = EvidenceEvaluator.from_defaults(context)
# answer_evaluator = AnswerEvaluator.from_defaults(context)
# hit_evaluator = HitEvaluator.from_defaults(context)
#
# for paper in tqdm(papers[:10]):
#     paper_index = PaperIndex(paper, context)
#     query_engine = paper_index.as_index().as_query_engine()
#     # evidence_df = evidence_evaluator.evaluate(paper, query_engine)
#     # hit_df = hit_evaluator.evaluate(paper, query_engine)
#     # join the two dataframes on the question column
#     # dataframe = pd.merge(evidence_df, hit_df, on="question")
#     dataframe = answer_evaluator.evaluate(paper, query_engine)
#     if scores_64.empty:
#         scores_64 = dataframe
#     else:
#         scores_64 = pd.concat([scores_64, dataframe])
#     del paper_index


# scores_512.to_csv("chunks_512.csv")
# scores_1024.to_csv("chunks_1024.csv")
# scores_64.to_csv("chunks_64.csv")

# scores_512.to_csv("answer_chunks_512.csv")
scores_1024.to_csv("answer_chunks_1024.csv")
# scores_64.to_csv("answer_chunks_64.csv")
