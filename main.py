import os
from typing import List

import streamlit as st
from llama_index import VectorStoreIndex, set_global_service_context, ServiceContext, set_global_handler, \
    SimpleDirectoryReader, PromptHelper, LLMPredictor
from llama_index.chat_engine.types import ChatMode
from llama_index.llms import ChatMessage, MessageRole
from llama_index.node_parser import SentenceSplitter
from llama_index.postprocessor import SentenceTransformerRerank


@st.cache_resource(show_spinner=False)
def load_context(enable_trace=True):
    with st.spinner("Loading context and models..."):
        if enable_trace:
            import phoenix as px
            px.launch_app()
            set_global_handler("arize_phoenix")

        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        prompt_helper = PromptHelper(
            context_window=4096,
            num_output=256,
            chunk_overlap_ratio=0.1,
            chunk_size_limit=None,
        )
        llm_predictor = LLMPredictor(llm="local")
        context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                               embed_model="local",
                                               text_splitter=text_splitter,
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
        st.sidebar.markdown("This app is a demo of the Paper Assistant.")

        if not os.path.exists(App.FILE_DIR):
            os.makedirs(App.FILE_DIR)

        # Create an file uploader for user to upload the data

        self.uploaded_file = st.sidebar.file_uploader("Upload your paper", type=App.FILE_FORMATS)

        st.sidebar.button("Load File", on_click=self.upload, disabled=self.uploaded_file is None, type="primary")

        st.header("📚Chat with your paper 💬")

        with st.chat_message(MessageRole.SYSTEM):
            st.write("Welcome to the Paper Assistant! Upload your paper on the left to get started.")

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
                    documents = SimpleDirectoryReader(input_files=[f"{App.FILE_DIR}/{self.uploaded_file.name}"]).load_data()
                else:
                    # Otherwise, use simple chat mode (no context)
                    st.info("No paper uploaded, using simple chat mode")
                    mode = ChatMode.SIMPLE
                    documents = []
                index = VectorStoreIndex.from_documents(documents, show_progress=True)
                response = index.as_chat_engine(
                    chat_mode=mode,
                    verbose=True,
                    similarity_top_k=5,
                    node_postprocessors=[
                        SentenceTransformerRerank(
                            model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=2,
                        ),
                    ],
                ).chat(prompt, chat_history=self.chat_history)
                with st.chat_message(MessageRole.ASSISTANT):
                    st.write(str(response))


if __name__ == "__main__":
    load_context()
    app = App()
