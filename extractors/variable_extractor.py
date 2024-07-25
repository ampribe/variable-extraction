"""
Module includes VariableExtractor class
VariableExtractor is a base class for variable extraction from an individual court case
"""
import json
from dataclasses import asdict
from abc import ABC, abstractmethod
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from utils.case_metadata import CaseMetadata
from extractors.extractor_config import ExtractorConfig

class VariableExtractor(ABC):
    """
    Provides base class for extracting variables from case (specified by metadata path)
    Each subclass must implement _get_default_config
    Public Methods:
        extract: extracts variable from case

    """
    def __init__(self, metadata: CaseMetadata, config: ExtractorConfig|None = None) -> None:
        """
        parameters:
            metadata: CaseMetadata of case to extract
            config: ExtractorConfig for this extractor
        """
        self.metadata = metadata
        if config is None:
            self.config = self._get_default_config(metadata)
        else:
            self.config = config
        self.log = self.metadata.get_metadata_json(
            fields=("court","title","docket","judges","link")
                ) | asdict(self.config)

    @staticmethod
    @abstractmethod
    def _get_default_config(metadata: CaseMetadata) -> ExtractorConfig:
        pass

    @classmethod
    def from_metadata_path(cls, path: str, config: ExtractorConfig|None=None)->"VariableExtractor":
        """
        Initializes VariableExtractor from path to metadata
        """
        metadata = CaseMetadata.from_metadata_path(path)
        return cls(metadata, config)

    def _chunk_text_list(self, text_list: list[str]) -> list[str]:
        """
        chunks list of texts
        uses separators, chunk size, chunk overlap from config
        """
        text_splitter = RecursiveCharacterTextSplitter(
            separators=self.config.separators,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.overlap,
        )
        return [
            chunk
            for text in text_list
            for chunk in text_splitter.split_text(text)
            ]

    def _filter_by_keyword_function(self, text_list: list[str]) -> list[str]:
        """
        filters text strings by provided content_filter function in config
        """
        return [
            text
            for text in text_list
            if self.config.content_filter(text)
            ]
        
    def _filter_by_semantic_similarity(self, text_list: list[str]) -> list[str]:
        """
        Constructs vector database from provided texts,
        filters documents by semantic similarity to embedding_model_prompt
        """
        client = chromadb.Client()
        collection = client.create_collection("case")
        for i, text in enumerate(text_list):
            if len(text) > 0:
                embedding = ollama.embeddings(
                    model=self.config.embedding_model,
                    prompt=text
                    )["embedding"]
                collection.add(ids=[str(i)], embeddings=[embedding], documents=[text])
        prompt_embedding = ollama.embeddings(
            model=self.config.embedding_model,
            prompt=self.config.embedding_model_prompt
            )["embedding"]
        return collection.query(
            query_embeddings=[prompt_embedding],
            n_results=self.config.llm_document_count
            )["documents"][0]

    def _get_relevant_chunks(self, docs: list[str]) -> list[str]:
        """
        Applies _chunk_text_list, _filter_by_keyword_function,
        _filter_by_semantic_similarity in order to text list
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
        context = self.config.document_separator.join(context_documents)
        response = ollama.generate(
            model=self.config.language_model,
            prompt=context,
            system=self.config.language_model_prompt,
            options={"num_ctx": self.config.llm_context_length},
            format="json",
        )
        return (response, context)

    def _extract_from_metadata(self) -> tuple[dict[str, str], str]:
        """
        queries llm with docket_report content and logs query
        returns response as json
        """
        print("Extracting from metadata...")
        print("- Getting relevant chunks...")
        relevant_chunks = self._get_relevant_chunks(self.metadata.get_docket_report_contents())
        if len(relevant_chunks) == 0:
            return (
                {self.config.variable_tag: self.config.null_value},
                "No relevant docket_report entries",
            )
        print("- Querying llm...")
        response, context = self._query_llm(relevant_chunks)
        try:
            response_json = json.loads(response["response"]) # pylint: disable=unsubscriptable-object
        except json.JSONDecodeError:
            response_json = {self.config.variable_tag: self.config.null_value,
                             "Response": str(response), "Status": "Invalid JSON response"}
        self.log["metadata_response"] = response
        self.log["metadata_response_json"] = response_json
        self.log["metadata_context"] = context
        return response_json


    def _extract_from_documents(self) -> tuple[dict[str, str], str]:
        """
        queries llm with relevant chunks from documents,
        only uses documents associated with relevant docket_report entry

        returns:
            response as json
        """
        print("Extracting from documents...")
        print("- Getting relevant chunks...")
        relevant_docs = self.metadata.get_documents_by_docket_report_filter(self.config.title_filter).values()
        relevant_chunks = self._get_relevant_chunks(relevant_docs)
        if len(relevant_chunks) == 0:
            return ({self.config.variable_tag: self.config.null_value}, "No relevant documents")
        print("- Querying llm...")
        response, context = self._query_llm(relevant_chunks)
        try:
            response_json = json.loads(response["response"]) # pylint: disable=unsubscriptable-object
        except json.JSONDecodeError:
            response_json = {self.config.variable_tag: self.config.null_value,
                             "Response": str(response), "Status": "Invalid JSON response"}
        self.log["document_response"] = response
        self.log["document_response_json"] = response_json
        self.log["document_context"] = context
        return response_json

    def extract(self) -> str:
        """
        first checks metadata for variable, then checks documents

        return string response
        """
        metadata_resp = self._extract_from_metadata()
        print(f"- Response: {metadata_resp}")
        if self.config.variable_tag in metadata_resp and metadata_resp[self.config.variable_tag]!=self.config.null_value:
            self.log["category"] = metadata_resp[self.config.variable_tag]
            return metadata_resp[self.config.variable_tag]
        document_resp = self._extract_from_documents()
        print(f"- Response: {document_resp}")
        if self.config.variable_tag in document_resp:
            self.log["category"] = document_resp[self.config.variable_tag]
            return document_resp[self.config.variable_tag]
        self.log["category"] = self.config.null_value
        return self.config.null_value
