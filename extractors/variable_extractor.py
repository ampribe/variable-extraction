"""
Module includes VariableExtractor class
"""
import json
from typing import Callable
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from utils.case_metadata import CaseMetadata

class VariableExtractor:
    """
    Provides base class for extracting variables from case (specified by metadata path)
    Public Methods:
        extract: extracts variable from case
        summarize: summarizes provided document

    """
    def __init__(
        self,
        metadata_path: str,
        prompt: str,
        variable_tag: str,
        null_value: str,
        content_filter: Callable[[str], bool],
        title_filter: Callable[[str], bool],
        embedding_model: str = "mxbai-embed-large",
        separators: list[str]|None = None,
        chunk_size: int = 700,
        overlap: int = 0,
        language_model: str = "mistral",
        llm_document_count: int = 9,
        document_separator: str = "||",
        llm_context_length: int = 2048,
    ) -> None:
        """
        parameters:
            metadata_path: path to metadata.json (assumes docs stored in the same parent directory)
            prompt: instructions for llm to evaluate provided documents,
                includes desired json formatting (eg. {reasoning: ..., variable: ...})
            variable_tag: name of variable of interest in json returned by llm eg. "variable"
            null_value: value for variable in json that indicates extraction failed
                (used to query documents if metadata query fails) eg. "unknown"
            content_filter: function used to filter content of text chunks
            title_filter: function used to filter documents based on title in docket_report
            embedding_model: model used to generate prompt and text embedding,
                models available: https://ollama.com/blog/embedding-models
            separators: separators used to chunk text in order of preference
            chunk_size: maximum size of each text chunk (in characters)
            overlap: overlap between chunks (in characters)
            language_model: model used to evaluate prompt,
                models available: https://ollama.com/library
            llm_document_count: number of documents to pass to llm
            document_separator: string used to separate documents passed to llm
            llm_context_length: length of llm context window (in tokens),
                must cover both system prompt and provided documents
        note metadata stored as CaseMetadata
        """
        self.metadata = CaseMetadata.from_metadata_path(metadata_path)
        self.prompt = prompt
        self.variable_tag = variable_tag
        self.null_value = null_value
        self.content_filter = content_filter
        self.title_filter = title_filter
        self.embedding_model = embedding_model
        self.separators = [
            "\n\n",
            "\n",
            ". ",
            "!",
            "?",
            ".",
            ";",
            ":",
            ",",
            " ",
            "",
        ] if separators is None else separators
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.language_model = language_model
        self.llm_document_count = llm_document_count
        self.document_separator = document_separator
        self.llm_context_length = llm_context_length

    def _chunk_text_list(self, text_list: list[str]) -> list[str]:
        """
        chunks list of texts using parameters given to constructor
        """
        text_splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
        )
        components = []
        for text in text_list:
            components += text_splitter.split_text(text)
        return components

    def _filter_by_keyword_function(self, text_list: list[str]) -> list[str]:
        """
        filters texts by provided content filtering function
        """
        return [text for text in text_list if self.content_filter(text)]

    def _filter_by_semantic_similarity(self, text_list: list[str]) -> list[str]:
        """
        Adds each document to vector database and returns documents most similar
        to given prompt
        """
        client = chromadb.Client()
        for collection in client.list_collections():
            if collection.name == "case":
                client.delete_collection("case")
        collection = client.create_collection("case")
        for i, text in enumerate(text_list):
            if len(text) > 0:
                embedding = ollama.embeddings(model=self.embedding_model, prompt=text)[
                    "embedding"
                ]
                collection.add(ids=[str(i)], embeddings=[embedding], documents=[text])
        prompt_embedding = ollama.embeddings(
            model=self.embedding_model, prompt=self.prompt
        )["embedding"]
        return collection.query(
            query_embeddings=[prompt_embedding], n_results=self.llm_document_count
        )["documents"][0]

    def _get_relevant_chunks(self, docs: list[str]) -> list[str]:
        """
        Applies chunking, then keyword filter,
        then semantic similarity filter to given document list
        """
        return self._filter_by_semantic_similarity(
            self._filter_by_keyword_function(self._chunk_text_list(docs))
        )

    def _query_llm(self, context_documents: list[str]) -> tuple[dict[str, str], str]:
        """
        queries llm based on query and documents given as context
        query (from parameters) given as system prompt, documents given as prompt
        returns tuple of response (after converting to json) and context provided from documents
        """
        context = self.document_separator.join(context_documents)
        response = ollama.generate(
            model=self.language_model,
            prompt=context,
            system=self.prompt,
            options={"num_ctx": self.llm_context_length},
            format="json",
        )
        return (json.loads(response["response"]), context)

    def _extract_from_metadata(self) -> tuple[dict[str, str], str]:
        """
        queries llm with docket_report content
        returns tuple of response and documents provided as context
        """
        print("Extracting from metadata...")
        print("- Getting relevant chunks...")
        relevant_chunks = self._get_relevant_chunks(self.metadata.get_docket_report_contents())
        if len(relevant_chunks) == 0:
            return (
                {self.variable_tag: self.null_value},
                "No relevant docket_report entries",
            )
        print("- Querying llm...")
        return self._query_llm(relevant_chunks)

    def _extract_from_documents(self) -> tuple[dict[str, str], str]:
        """
        queries llm with relevant chunks from documents,
        only uses documents associated with relevant docket_report entry

        returns:
            tuple of response and documents provided as context
        """
        print("Extracting from documents...")
        print("- Getting relevant chunks...")
        relevant_docs = self.metadata.get_documents_by_docket_report(self.title_filter)
        relevant_chunks = self._get_relevant_chunks(relevant_docs)
        if len(relevant_chunks) == 0:
            return ({self.variable_tag: self.null_value}, "No relevant documents")
        print("- Querying llm...")
        return self._query_llm(relevant_chunks)

    def extract(self) -> tuple[dict[str, str], dict[str, str]]:
        """
        first checks metadata for variable, then checks documents

        return tuple of response and log
        """
        log = {}
        metadata_resp, metadata_context = self._extract_from_metadata()
        print(f"- Response: {metadata_resp}")
        log["metadata_response"] = metadata_resp
        log["metadata_context"] = metadata_context
        if metadata_resp[self.variable_tag] != self.null_value:
            return (metadata_resp, log)
        document_resp, document_context = self._extract_from_documents()
        log["document_response"] = document_resp
        log["document_context"] = document_context
        print(f"- Response: {document_resp}")
        return (document_resp, log)

    def summarize(self, document) -> str:
        """
        summarizes legal document
        """
        system_prompt = """
        You are an expert legal analyst. The user message will contain a sequence of documents related to a legal case in the United States.
        Your task is to summarize the events of the case. Describe each document in 10 words or less.
        Documents may contain metadata such as filing date or the name of the person issuing the document. These are not important, do not include them in your answer.
        Documents may include extra symbols or numbers that do not describe what the document is. Do not include these in your answer.
        Do not summarize each document. Only summarize the major events during the case. If multiple documents describe redundant information, combine them into one description.
        """
        return ollama.generate(
            model=self.language_model,
            prompt=document,
            system=system_prompt,
            options={"num_ctx": self.llm_context_length},
        )["response"]
