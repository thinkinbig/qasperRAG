from llama_index import PromptTemplate
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL, DEFAULT_REFINE_PROMPT_TMPL

TEXT_QA_SYSTEM_PROMPT_TMPL = ("<s> [INST] <<SYS>> "
                              "You are an expert Q&A system that is trusted around the world.\n"
                              "Always answer the query using the provided context information, "
                              "and not prior knowledge.\n"
                              "Some rules to follow:\n"
                              "1.Answer the question in extractive manner.\n"
                              "2.Address the question directly with most relevant part.\n"
                              "<</SYS>> [/INST] </s>")

DEFAULT_LLAMA_TEXT_QA_PROMPT = PromptTemplate(TEXT_QA_SYSTEM_PROMPT_TMPL + DEFAULT_TEXT_QA_PROMPT_TMPL)
DEFAULT_LLAMA_REFINE_PROMPT = PromptTemplate(TEXT_QA_SYSTEM_PROMPT_TMPL + DEFAULT_REFINE_PROMPT_TMPL)
