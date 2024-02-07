# Read dataset from disk
import pandas as pd

chunk_64 = pd.read_csv("chunks_64.csv")

chunk_512 = pd.read_csv("chunks_512.csv")

chunk_1024 = pd.read_csv("chunks_1024.csv")

baseline_rerank = pd.read_csv("baseline_rerank.csv")

finetuned_rerank = pd.read_csv("finetune_rerank.csv")

finetuned_finetuned_rerank = pd.read_csv("finetuned_finetuned_rerank.csv")

finetuned_emb = pd.read_csv("finetuned_emb.csv")

bm_25 = pd.read_csv("bm25_retriever.csv")


df_result_table = pd.DataFrame(columns=["model", "evidence-f1", "context-recall", "context-precision"])

df_result_table["model"] = ["chunk size 64", "chunk size 512", "chunk size 1024", "baseline emb + rerank", "finetuned emb + rerank",
                            "finetuned emb", "finetuned emb + finetuned rerank",
                            "bm25"]

# Calculate the average evidence F1 score, context recall and context precision
# ignore 0
chunk_64_avg = chunk_64["evidence-f1"].mean()
chunk_512_avg = chunk_512["evidence-f1"].mean()
chunk_1024_avg = chunk_1024["evidence-f1"].mean()
baseline_rerank_f1_avg = baseline_rerank["evidence-f1"].mean()
finetuned_rerank_f1_avg = finetuned_rerank["evidence-f1"].mean()
finetuned_emb_f1_avg = finetuned_emb["evidence-f1"].mean()
finetuned_finetuned_rerank_f1_avg = finetuned_finetuned_rerank["evidence-f1"].mean()
bm_25_f1_avg = bm_25["evidence-f1"].mean()

df_result_table["evidence-f1"] = [chunk_64_avg, chunk_512_avg, chunk_1024_avg, baseline_rerank_f1_avg,
                                  finetuned_rerank_f1_avg, finetuned_emb_f1_avg,
                                  finetuned_finetuned_rerank_f1_avg, bm_25_f1_avg]

print(f"Average evidence F1 score for chunk size 64: {chunk_64_avg}")

print(f"Average evidence F1 score for chunk size 512: {chunk_512_avg}")

print(f"Average evidence F1 score for chunk size 1024: {chunk_1024_avg}")

print(f"Average evidence F1 score for baseline rerank: {baseline_rerank_f1_avg}")

print(f"Average evidence F1 score for finetuned emb + rerank: {finetuned_rerank_f1_avg}")

print(f"Average evidence F1 score for finetuned emb: {finetuned_emb_f1_avg}")

print(f"Average evidence F1 score for finetuned emb + finetuned rerank: {finetuned_finetuned_rerank_f1_avg}")

print(f"Average evidence F1 score for bm25: {bm_25_f1_avg}")


context_recall_64_avg = chunk_64["context-recall"].mean()
context_recall_512_avg = chunk_512["context-recall"].mean()
context_recall_1024_avg = chunk_1024["context-recall"].mean()
baseline_rerank_recall_avg = baseline_rerank["context-recall"].mean()
finetuned_rerank_recall_avg = finetuned_rerank["context-recall"].mean()
finetuned_emb_recall_avg = finetuned_emb["context-recall"].mean()
fine_tuned_finetuned_rerank_recall_avg = finetuned_finetuned_rerank["context-recall"].mean()
bm_25_recall_avg = bm_25["context-recall"].mean()

df_result_table["context-recall"] = [context_recall_64_avg, context_recall_512_avg, context_recall_1024_avg,
                                     baseline_rerank_recall_avg, finetuned_rerank_recall_avg,
                                     fine_tuned_finetuned_rerank_recall_avg,
                                     finetuned_emb_recall_avg, bm_25_recall_avg]

print(f"Average context recall for chunk size 64: {context_recall_64_avg}")

print(f"Average context recall for chunk size 512: {context_recall_512_avg}")

print(f"Average context recall for chunk size 1024: {context_recall_1024_avg}")

print(f"Average context recall for baseline rerank: {baseline_rerank_recall_avg}")

print(f"Average context recall for finetuned rerank: {finetuned_rerank_recall_avg}")

print(f"Average context recall for finetuned emb: {finetuned_emb_recall_avg}")

print(f"Average context recall for finetuned emb + finetuned rerank: {fine_tuned_finetuned_rerank_recall_avg}")

print(f"Average context recall for bm25: {bm_25_recall_avg}")


context_precision_64_avg = chunk_64["context-precision"].mean()
context_precision_512_avg = chunk_512["context-precision"].mean()
context_precision_1024_avg = chunk_1024["context-precision"].mean()
baseline_rerank_precision_avg = baseline_rerank["context-precision"].mean()
finetuned_rerank_precision_avg = finetuned_rerank["context-precision"].mean()
finetuned_emb_precision_avg = finetuned_emb["context-precision"].mean()
finetuned_finetuned_rerank_precision_avg = finetuned_finetuned_rerank["context-precision"].mean()
bm_25_precision_avg = bm_25["context-precision"].mean()

df_result_table["context-precision"] = [context_precision_64_avg, context_precision_512_avg, context_precision_1024_avg,
                                        baseline_rerank_precision_avg, finetuned_emb_precision_avg,
                                        finetuned_finetuned_rerank_precision_avg,
                                        finetuned_emb_precision_avg, bm_25_precision_avg]

print(f"Average context precision for chunk size 64: {context_precision_64_avg}")

print(f"Average context precision for chunk size 512: {context_precision_512_avg}")

print(f"Average context precision for chunk size 1024: {context_precision_1024_avg}")

print(f"Average context precision for baseline rerank: {baseline_rerank_precision_avg}")

print(f"Average context precision for finetuned rerank: {finetuned_rerank_precision_avg}")

print(f"Average context precision for finetuned emb: {finetuned_emb_precision_avg}")

print(f"Average context precision for finetuned emb + finetuned rerank: {finetuned_finetuned_rerank_precision_avg}")

print(f"Average context precision for bm25: {bm_25_precision_avg}")

# chunk_64_hit_avg = chunk_64["hit"].mean()
# chunk_512_hit_avg = chunk_512["hit"].mean()
# chunk_1024_hit_avg = chunk_1024["hit"].mean()
# baseline_rerank_hit_avg = baseline_rerank["hit"].mean()
# finetuned_emb_hit_avg = finetuned_emb["hit"].mean()
# bm_25_hit_avg = bm_25["hit"].mean()
# window_parser_hit_avg = window_parser["hit"].mean()
#
# df_result_table["hit"] = [chunk_64_hit_avg, chunk_512_hit_avg, chunk_1024_hit_avg, baseline_rerank_hit_avg,
#                           finetuned_emb_hit_avg, bm_25_hit_avg, window_parser_hit_avg]
#
# print(f"Average hit for chunk size 64: {chunk_64_hit_avg}")
#
# print(f"Average hit for chunk size 512: {chunk_512_hit_avg}")
#
# print(f"Average hit for chunk size 1024: {chunk_1024_hit_avg}")
#
# print(f"Average hit for baseline rerank: {baseline_rerank_hit_avg}")
#
# print(f"Average hit for finetuned emb: {finetuned_emb_hit_avg}")
#
# print(f"Average hit for bm25: {bm_25_hit_avg}")

# print(f"Average hit for window parser: {window_parser_hit_avg}")

df_result_table.to_csv("evidence_result_table.csv", index=False)

# Plot the results table
import matplotlib.pyplot as plt
import numpy as np

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = df_result_table["evidence-f1"]
bars2 = df_result_table["context-recall"]
bars3 = df_result_table["context-precision"]
# bars4 = df_result_table["hit"]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.figure(figsize=(10, 5))

# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='evidence-f1')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='context-recall')
plt.bar(r3, bars3, color='#3d7f5e', width=barWidth, edgecolor='white', label='context-precision')
# plt.bar(r4, bars4, color='#e2b007', width=barWidth, edgecolor='white', label='hit')

# Add xticks on the middle of the group bars
plt.xlabel('Model', fontweight='bold')

plt.ylabel('Score', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(bars1))], df_result_table["model"])

# Create legend & Show graphic
plt.legend()

plt.savefig("result_table.png")

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.title("Evidence F1 score for different models")

plt.plot(chunk_64["evidence-f1"], label="chunk size 64")
plt.plot(chunk_512["evidence-f1"], label="chunk size 512")
plt.plot(chunk_1024["evidence-f1"], label="chunk size 1024")
# plt.plot(baseline_rerank["evidence-f1"], label="baseline rerank")
# plt.plot(finetuned_emb["evidence-f1"], label="finetuned emb")
# plt.plot(bm_25["evidence-f1"], label="bm25")
# plt.plot(window_parser["evidence-f1"], label="window parser")

plt.axhline(y=chunk_64_avg, color="r", linestyle="-", label="chunk size 64 average")

plt.axhline(y=chunk_512_avg, color="g", linestyle="-", label="chunk size 512 average")

plt.axhline(y=chunk_1024_avg, color="b", linestyle="-", label="chunk size 1024 average")

# plt.axhline(y=baseline_rerank_f1_avg, color="y", linestyle="-", label="baseline rerank average")
#
# plt.axhline(y=finetuned_emb_f1_avg, color="m", linestyle="-", label="finetuned emb average")
#
# plt.axhline(y=bm_25_f1_avg, color="c", linestyle="-", label="bm25 average")
#
# plt.axhline(y=window_parser_f1_avg, color="k", linestyle="-", label="window parser average")

plt.xlabel("Question")

plt.ylabel("Evidence F1")

plt.legend()

plt.savefig("chunks_evidence_f1.png")

# plt.savefig("models_evidence_f1.png")

# Plot the Context Recall

plt.figure(figsize=(10, 5))

plt.title("Context Recall for different models")

plt.plot(chunk_64["context-recall"], label="chunk size 64")

plt.plot(chunk_512["context-recall"], label="chunk size 512")

plt.plot(chunk_1024["context-recall"], label="chunk size 1024")

# plt.plot(baseline_rerank["context-recall"], label="baseline rerank")
#
# plt.plot(finetuned_emb["context-recall"], label="finetuned emb")
#
# plt.plot(bm_25["context-recall"], label="bm25")
#
# plt.plot(window_parser["context-recall"], label="window parser")

plt.axhline(y=context_recall_64_avg, color="r", linestyle="-", label="chunk size 64 average")

plt.axhline(y=context_recall_512_avg, color="g", linestyle="-", label="chunk size 512 average")

plt.axhline(y=context_recall_1024_avg, color="b", linestyle="-", label="chunk size 1024 average")

# plt.axhline(y=baseline_rerank_recall_avg, color="y", linestyle="-", label="baseline rerank average")
#
# plt.axhline(y=finetuned_emb_recall_avg, color="m", linestyle="-", label="finetuned emb average")
#
# plt.axhline(y=bm_25_recall_avg, color="c", linestyle="-", label="bm25 average")
#
# plt.axhline(y=window_parser_recall_avg, color="k", linestyle="-", label="window parser average")

plt.xlabel("Question")

plt.ylabel("Context Recall")

plt.legend()

# plt.savefig("models_context_recall.png")

plt.savefig("chunks_context_recall.png")

# Plot the Context Precision

plt.figure(figsize=(10, 5))

plt.title("Context Precision for different models")

plt.plot(chunk_64["context-precision"], label="chunk size 64")

plt.plot(chunk_512["context-precision"], label="chunk size 512")

plt.plot(chunk_1024["context-precision"], label="chunk size 1024")

# plt.plot(baseline_rerank["context-precision"], label="baseline rerank")
#
# plt.plot(finetuned_emb["context-precision"], label="finetuned emb")
#
# plt.plot(bm_25["context-precision"], label="bm25")
#
# plt.plot(window_parser["context-precision"], label="window parser")

plt.axhline(y=context_precision_64_avg, color="r", linestyle="-", label="chunk size 64 average")

plt.axhline(y=context_precision_512_avg, color="g", linestyle="-", label="chunk size 512 average")

plt.axhline(y=context_precision_1024_avg, color="b", linestyle="-", label="chunk size 1024 average")

# plt.axhline(y=baseline_rerank_precision_avg, color="y", linestyle="-", label="baseline rerank average")
#
# plt.axhline(y=finetuned_emb_precision_avg, color="m", linestyle="-", label="finetuned emb average")
#
# plt.axhline(y=bm_25_precision_avg, color="c", linestyle="-", label="bm25 average")
#
# plt.axhline(y=window_parser_precision_avg, color="k", linestyle="-", label="window parser average")

plt.xlabel("Question")

plt.ylabel("Context Precision")

plt.legend()

# plt.savefig("models_context_precision.png")

plt.savefig("chunks_context_precision.png")

# Plot the Hit

# plt.figure(figsize=(10, 5))
#
# plt.title("Hit for different models")
#
# plt.plot(chunk_64["hit"], label="chunk size 64")
#
# plt.plot(chunk_512["hit"], label="chunk size 512")
#
# plt.plot(chunk_1024["hit"], label="chunk size 1024")

# plt.plot(baseline_rerank["hit"], label="baseline rerank")
#
# plt.plot(finetuned_emb["hit"], label="finetuned emb")
#
# plt.plot(bm_25["hit"], label="bm25")
#
# plt.plot(window_parser["hit"], label="window parser")

# plt.axhline(y=chunk_64_hit_avg, color="r", linestyle="-", label="chunk size 64 average")
#
# plt.axhline(y=chunk_512_hit_avg, color="g", linestyle="-", label="chunk size 512 average")
#
# plt.axhline(y=chunk_1024_hit_avg, color="b", linestyle="-", label="chunk size 1024 average")

# plt.axhline(y=baseline_rerank_hit_avg, color="y", linestyle="-", label="baseline rerank average")
#
# plt.axhline(y=finetuned_emb_hit_avg, color="m", linestyle="-", label="finetuned emb average")
#
# plt.axhline(y=bm_25_hit_avg, color="c", linestyle="-", label="bm25 average")
#
# plt.axhline(y=window_parser_hit_avg, color="k", linestyle="-", label="window parser average")
#
# plt.xlabel("Question")
#
# plt.ylabel("Hit")
#
# plt.legend()
#
# plt.savefig("models_hit.png")
#
# plt.savefig("chunks_hit.png")
