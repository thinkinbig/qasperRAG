import os
from typing import List

import streamlit as st
from llama_index import VectorStoreIndex, set_global_service_context, ServiceContext, set_global_handler, \
    SimpleDirectoryReader, PromptHelper, LLMPredictor
from llama_index.chat_engine.types import ChatMode
from llama_index.embeddings import resolve_embed_model
from llama_index.llms import ChatMessage, MessageRole
from llama_index.node_parser import SentenceSplitter
from llama_index.postprocessor import SentenceTransformerRerank

from qasper_data.qasper_prompt import TEXT_QA_SYSTEM_PROMPT_TMPL
from transformations.ASRTextCleaner import ASRTextCleaner




@st.cache_resource(show_spinner=False)
def load_context(enable_trace=True):
    with st.spinner("Loading context and models..."):
        if enable_trace:
            import phoenix as px
            px.launch_app()
            # set_global_handler("arize_phoenix")
            set_global_handler("simple")

        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=0, paragraph_separator="\n")
        llm_predictor = LLMPredictor(llm="local")
        prompt_helper = PromptHelper.from_llm_metadata(llm_predictor.metadata)
        asr_text_cleaner = ASRTextCleaner()
        transformations = [
            text_splitter,
            asr_text_cleaner,
            # HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
            resolve_embed_model("local:models/fine_tuned_embedding")
        ]
        context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                               embed_model=resolve_embed_model("local:models/fine_tuned_embedding"),
                                               transformations=transformations,
                                               prompt_helper=prompt_helper)
        set_global_service_context(context)


class App:
    uploaded_file = None
    index = None

    FILE_FORMATS = ["pdf", "txt"]
    FILE_DIR = "data"

    def upload(self):
        if self.uploaded_file is None:
            st.error("Please upload a file")
            return
        # if the size of the uploaded file is greater than 200MB then throw an error
        if self.uploaded_file.size > 200000000:
            st.error("Please upload a file smaller than 200MB")
            return
        # clear the file directory
        assert os.path.isdir(App.FILE_DIR)
        for file in os.listdir(App.FILE_DIR):
            os.remove(os.path.join(App.FILE_DIR, file))
        # save the uploaded file to the file directory
        with open(f"{App.FILE_DIR}/{self.uploaded_file.name}", "wb") as f:
            f.write(self.uploaded_file.getbuffer())
        st.success(f"File uploaded successfully: {self.uploaded_file.name}")

    @property
    def chat_history(self) -> List[ChatMessage]:
        return st.session_state.messages

    def clear_chat_history(self):
        self.chat_history.clear()

    def __init__(self):
        st.title("Paper Chatbot Assistant")
        st.sidebar.title("Upload your paper")

        if not os.path.exists(App.FILE_DIR):
            os.makedirs(App.FILE_DIR)

        # Create an file uploader for user to upload the data

        self.uploaded_file = st.sidebar.file_uploader("Upload your paper", type=App.FILE_FORMATS)

        st.sidebar.button("Load File", on_click=self.upload, disabled=self.uploaded_file is None, type="primary")

        st.header("📚Chat with your lecture paper 💬")

        with st.chat_message(MessageRole.SYSTEM):
            st.write("Welcome to the Lecture Assistant! Upload your lecture paper on the left to get started.")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for chat_message in st.session_state.messages:
            with st.chat_message(chat_message.role):
                st.write(chat_message.content)

        # Accept user input
        if prompt := st.chat_input("Type your question here..."):
            # Display user message in chat message container
            assert len(self.chat_history) == 0 or self.chat_history[-1].role == MessageRole.ASSISTANT
            with st.chat_message(MessageRole.USER):
                st.write(prompt)
            with st.spinner("Thinking..., this may take a while"):
                if self.uploaded_file is not None:
                    # Use context chat mode if the user's message is a question
                    mode = ChatMode.CONTEXT
                    documents = SimpleDirectoryReader(
                        input_files=[f"{App.FILE_DIR}/{self.uploaded_file.name}"]).load_data()
                else:
                    # Otherwise, use simple chat mode (no context)
                    st.info("No paper uploaded, using simple chat mode")
                    mode = ChatMode.SIMPLE
                    documents = []
                index = VectorStoreIndex.from_documents(documents, show_progress=True)
                response = index.as_chat_engine(
                    chat_mode=mode,
                    verbose=True,
                    similarity_top_k=8,
                    node_postprocessors=[
                        SentenceTransformerRerank(
                            model="models/fine_tuned_rerank", top_n=3,
                        ),
                    ],
                    system_prompt=TEXT_QA_SYSTEM_PROMPT_TMPL,
                ).chat(prompt, chat_history=self.chat_history)
                with st.chat_message(MessageRole.ASSISTANT):
                    st.write(str(response))


if __name__ == "__main__":
    load_context()
    app = App()
