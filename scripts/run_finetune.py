import random

from llama_index import ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings import resolve_embed_model
from llama_index.finetuning.cross_encoders.cross_encoder import CrossEncoderFinetuneEngine
from tqdm import tqdm

from qasper_data.qasper_dataset import QasperDataset, PaperIndex
from qasper_data.qasper_finetune import MyEmbeddingFinetuneEngine, generate_qa_embedding_pairs, generate_qa_cross_encoder_pairs
from qasper_data.qasper_entity import PARAGRAPH_SEPARATOR

train_papers = QasperDataset("train").as_papers()
validation_papers = QasperDataset("validation").as_papers()

node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=0, paragraph_separator=PARAGRAPH_SEPARATOR)

service_context = ServiceContext.from_defaults(llm=None, embed_model="local", node_parser=node_parser)

model_id = "BAAI/bge-small-en"
model_output_path = "../models/fine_tuned_embedding_1"

for paper in tqdm(train_papers):
    paper_index = PaperIndex(paper, service_context)
    train_dataset = generate_qa_embedding_pairs(service_context, paper)
    fine_tune_engine = MyEmbeddingFinetuneEngine(
        model_id=model_id,
        model_output_path=model_output_path,
        dataset=train_dataset,
        batch_size=16,
        epochs=2,
        show_progress_bar=True,
    )

    fine_tune_engine.finetune()

    model_id = model_output_path



# Finetune the Rerank

service_context = ServiceContext.from_defaults(llm=None, embed_model=resolve_embed_model("local:../models"
                                                                                         "/fine_tuned_embedding"),
                                               node_parser=node_parser)

model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"

model_output_path = "../models/fine_tuned_rerank"


for paper in tqdm(train_papers):
    train_dataset = generate_qa_cross_encoder_pairs(service_context, paper)
    if len(train_dataset) == 0:
        continue
    validation_paper = random.choice(validation_papers)
    validation_dataset = generate_qa_cross_encoder_pairs(service_context, validation_paper)
    while len(validation_dataset) == 0:
        validation_paper = random.choice(validation_papers)
        validation_dataset = generate_qa_cross_encoder_pairs(service_context, validation_paper)
    fine_tune_engine = CrossEncoderFinetuneEngine(
        model_id=model_id,
        model_output_path=model_output_path,
        dataset=train_dataset,
        val_dataset=validation_dataset,
        batch_size=16,
        epochs=2,
        show_progress_bar=True,
    )
    fine_tune_engine.finetune()

    model_id = model_output_path


