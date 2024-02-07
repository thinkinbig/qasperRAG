from llama_index import ServiceContext, get_response_synthesizer
from llama_index.node_parser import SimpleNodeParser
from llama_index.retrievers import BM25Retriever

from qasper_data.oracle_retriever import OracleRetriever
from qasper_data.qasper_entity import PARAGRAPH_SEPARATOR
from qasper_data.qasper_dataset import QasperDataset, PaperIndex

from qasper_data.qasper_evaluator import EvidenceEvaluator, HitEvaluator, AnswerEvaluator, evaluate_response
from qasper_data.qasper_prompt import DEFAULT_LLAMA_TEXT_QA_PROMPT, DEFAULT_LLAMA_REFINE_PROMPT

from tqdm import tqdm

import pandas as pd

context = ServiceContext.from_defaults(llm="local", embed_model="local")

answer_evaluator = AnswerEvaluator.from_defaults(context)

papers = QasperDataset("test").as_papers()

# df_scores = pd.DataFrame(columns=["question", "evidence-f1", "context-recall", "context-precision"])
df_scores = pd.DataFrame(columns=["question", "answer-f1", "answer-recall", "answer-precision"])

for paper in tqdm(papers[:100]):
    paper_index = PaperIndex(paper, context)
    retriever = OracleRetriever.from_defaults(paper, context)
    # define response synthesizer
    response_synthesizer = get_response_synthesizer(service_context=context,
                                                    text_qa_template=DEFAULT_LLAMA_TEXT_QA_PROMPT,
                                                    refine_template=DEFAULT_LLAMA_REFINE_PROMPT,)
    # query_engine = paper_index.as_index().as_query_engine(
    #     retriever=retriever,
    #     response_synthesizer=response_synthesizer,
    # )
    # TODO: FIX query engine or evaluate answer by hand
    for question in paper.qas.question:
        nodes = retriever.retrieve(question)
        text_chunks = [node.get_content() for node in nodes]
        response = response_synthesizer.get_response(question, text_chunks)
        f1, recall, precision = evaluate_response(question, response)
        dataframe = pd.DataFrame([[question, f1, recall, precision]], columns=["question", "answer-f1", "answer-recall", "answer-precision"])
        if df_scores.empty:
            df_scores = dataframe
        else:
            df_scores = pd.concat([df_scores, dataframe])
    #
    # dataframe = evidence_evaluator.evaluate(paper, query_engine)
    # hit_df = hit_evaluator.evaluate(paper, query_engine)
    # join the two dataframes on the question column
    # dataframe = pd.merge(evidence_df, hit_df, on="question")
    # dataframe = answer_evaluator.evaluate(paper, query_engine)


# df_scores.to_csv("bm25_retriever.csv", index=False)
df_scores.to_csv("oracle_answers.csv", index=False)
