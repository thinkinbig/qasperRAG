import pandas as pd

df_result_table = pd.DataFrame(columns=["model", "answer-f1", "answer-recall", "answer-precision"])

df_result_table["model"] = ["baseline", "baseline emb + rerank", "finetuned emb + rerank",
                            "finetuned emb", "finetuned emb + finetuned rerank",
                            "bm25", "oracle"]

baseline = pd.read_csv("answer_chunks_512.csv")

baseline_rerank = pd.read_csv("baseline_rerank_answer.csv")

bm25 = pd.read_csv("bm25_retriever_answer.csv")

oracle = pd.read_csv("oracle_answers.csv")

finetuned_emb = pd.read_csv("finetuned_emb_answer.csv")

finetuned_finetuned_rerank = pd.read_csv("finetuned_finetuned_rerank_answers.csv")

finetuned_rerank = pd.read_csv("finetune_rerank_answers.csv")

baseline_avg = baseline["answer-f1"].mean()
baseline_rerank_avg = baseline_rerank["answer-f1"].mean()
bm25_avg = bm25["answer-f1"].mean()
finetuned_emb_avg = finetuned_emb["answer-f1"].mean()
finetuned_finetuned_rerank_avg = finetuned_finetuned_rerank["answer-f1"].mean()
finetuned_rerank_avg = finetuned_rerank["answer-f1"].mean()
oracle_avg = oracle["answer-f1"].mean()

df_result_table["answer-f1"] = [baseline_avg, baseline_rerank_avg, finetuned_rerank_avg,
                                finetuned_emb_avg, finetuned_finetuned_rerank_avg,
                                bm25_avg, oracle_avg]

baseline_avg = baseline["answer-recall"].mean()
baseline_rerank_avg = baseline_rerank["answer-recall"].mean()
bm25_avg = bm25["answer-recall"].mean()
finetuned_emb_avg = finetuned_emb["answer-recall"].mean()
finetuned_finetuned_rerank_avg = finetuned_finetuned_rerank["answer-recall"].mean()
finetuned_rerank_avg = finetuned_rerank["answer-recall"].mean()
oracle_avg = oracle["answer-recall"].mean()

df_result_table["answer-recall"] = [baseline_avg, baseline_rerank_avg, finetuned_rerank_avg,
                                    finetuned_emb_avg, finetuned_finetuned_rerank_avg,
                                    bm25_avg, oracle_avg]

baseline_avg = baseline["answer-precision"].mean()
baseline_rerank_avg = baseline_rerank["answer-precision"].mean()
bm25_avg = bm25["answer-precision"].mean()
finetuned_emb_avg = finetuned_emb["answer-precision"].mean()
finetuned_finetuned_rerank_avg = finetuned_finetuned_rerank["answer-precision"].mean()
finetuned_rerank_avg = finetuned_rerank["answer-precision"].mean()
oracle_avg = oracle["answer-precision"].mean()

df_result_table["answer-precision"] = [baseline_avg, baseline_rerank_avg, finetuned_rerank_avg,
                                       finetuned_emb_avg, finetuned_finetuned_rerank_avg,
                                       bm25_avg, oracle_avg]

df_result_table.to_csv("answer_result_table.csv", index=False)
