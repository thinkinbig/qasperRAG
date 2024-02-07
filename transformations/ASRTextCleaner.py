from typing import List

from llama_index.node_parser import SentenceSplitter
from llama_index.schema import TransformComponent, TextNode
import re


class ASRTextCleaner(TransformComponent):
    """
    A component that cleans the ASR text
    """

    @staticmethod
    def is_asr_segment(node: TextNode) -> bool:
        """
        Checks if the node is an ASR segment
        :param node: the node to check
        :return: True if the node is an ASR segment, False otherwise
        """
        return re.search("textseg:asr\d+: OUTPUT [0-9.]+-[0-9.]+: (.*)", node.text) is not None

    def __call__(self, nodes: List[TextNode], **kwargs):
        """
        Cleans the ASR text
        :param nodes: the nodes to clean
        :return: the cleaned nodes
        """
        for node in nodes:
            if self.is_asr_segment(node):
                asr_output = node.text
                speech_segments = re.findall(r"textseg:asr\d+: OUTPUT [0-9.]+-[0-9.]+: (.*)", asr_output)

                # Remove the HTML tags
                speech_segments = [re.sub(r"<.*?>", "", segment) for segment in speech_segments]

                # Remove the punctuation
                speech_segments = [re.sub(r"[^\w\s]", "", segment) for segment in speech_segments]

                # Remove the empty segments
                speech_segments = [segment for segment in speech_segments if len(segment) > 0]

                # concatenate the segments
                speech_segments = " ".join(speech_segments)

                # remove the extra spaces
                speech_segments = re.sub(r"\s+", " ", speech_segments)

                # Remove stop words like "um", "uh", "so" etc.
                from nltk.corpus import stopwords
                import nltk

                if not nltk.data.find("corpora/stopwords"):
                    nltk.download("stopwords")
                stop_words = stopwords.words("english") + ["um", "uh", "yeah", "Okay"]

                speech_segments = " ".join([word for word in speech_segments.split() if word not in stop_words])

                # remove the leading and trailing spaces
                speech_segments = speech_segments.strip()

                node.text = speech_segments
        return nodes