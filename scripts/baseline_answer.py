import pandas as pd
from llama_index import ServiceContext
from llama_index.node_parser import SimpleNodeParser
from tqdm import tqdm

df_answers = pd.DataFrame(columns=["question", "paper_id", "answer_type", "answer_string", "prediction"])


# evidence of the baseline model top_k=2, split by ;
from qasper_data.qasper_dataset import QasperDataset, PaperIndex
from qasper_data.qasper_entity import Paper, PARAGRAPH_SEPARATOR

papers = QasperDataset("test").as_papers()

chunk_512_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=0,
                                                  paragraph_separator=PARAGRAPH_SEPARATOR)

context = ServiceContext.from_defaults(llm="local", embed_model="local", node_parser=chunk_512_parser)

for paper in tqdm(papers[:10]):
    paper_index = PaperIndex(paper, context)
    query_engine = paper_index.as_index().as_query_engine()
    for question, answers in zip(paper.qas.question, paper.qas.answers):
        response = query_engine.query(question)
        # evidences = []
        # for node in response.source_nodes:
        #     evidences.append(node.get_content())
        # evidence = ";".join(evidences)
        for answer in answers.answer:
            df_answers = pd.concat([df_answers,
                                    pd.DataFrame([[question,
                                                   paper.id,
                                                   answer.answer_type,
                                                   answer.answer_string,
                                                   # evidence,
                                                   response.response]],
                                                 columns=["question",
                                                          "paper_id",
                                                          "answer_type",
                                                          "answer_string",
                                                          # "evidence",
                                                          "prediction"])])
    del paper_index

df_answers.to_csv("baseline_answers.csv", index=False)