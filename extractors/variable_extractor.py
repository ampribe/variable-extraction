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
from extractors.extractor_log import ExtractorLog

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
        self.config = config if config is not None else self._get_default_config(metadata)
        self.log = ExtractorLog(
            config=asdict(self.config),
            metadata=self.metadata.get_metadata_json(fields=("court","title","docket","judges","link")))

    @staticmethod
    @abstractmethod
    def _get_default_config(metadata: CaseMetadata) -> ExtractorConfig:
        """
        Generates default configuration for variable extractor 
        Implemented by subclass
        """

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
        for collection in client.list_collections():
            if collection.name == "case":
                client.delete_collection("case")
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

    def _get_relevant_chunks(self, docs: list[str]) -> tuple[list[str], dict[str, list[str]]]:
        """
        Applies _chunk_text_list, _filter_by_keyword_function,
        _filter_by_semantic_similarity in order to text list
        returns relevant chunks and log of filtering process
        """
        log = {}
        log["docs_before_chunking"] = docs
        log["chunks"] = self._chunk_text_list(log["docs_before_chunking"])
        log["chunks_after_keyword_filter"] = self._filter_by_keyword_function(log["chunks"])
        log["chunks_after_semantic_filter"] = self._filter_by_semantic_similarity(log["chunks_after_keyword_filter"])
        return (log["chunks_after_semantic_filter"], log)

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
        )["response"] # pylint: disable=unsubscriptable-object
        return (response, context)

    def _load_response(self, response) -> str:
        """
        Attempts to load json response from model
        Returns null tag if failure
        """
        try:
            return json.loads(response)[self.config.variable_tag] 
        except:
            return self.config.null_value

    def _extract_from_metadata(self) -> str:
        """
        queries llm with docket_report content and logs query
        returns response as json
        """
        print("Extracting from metadata...")
        print("- Getting relevant chunks...")
        chunks, log = self._get_relevant_chunks(self.metadata.get_docket_report_contents())
        log["llm_context"] = None
        log["llm_response_raw"] = None
        log["llm_response"] = self.config.null_value
        self.log.update_metadata_classification(log)
        if len(chunks) == 0:
            return self.config.null_value
        print("- Querying llm...")
        log["llm_response_raw"], log["llm_context"] = self._query_llm(chunks)
        log["llm_response"] = self._load_response(log["llm_response_raw"])
        self.log.update_metadata_classification(log)
        return log["llm_response"]


    def _extract_from_documents(self) -> str:
        """
        queries llm with relevant chunks from documents,
        only uses documents associated with relevant docket_report entry

        returns:
            response as json
        """
        print("Extracting from documents...")
        print("- Getting relevant chunks...")
        doc_dict = self.metadata.get_documents_by_docket_report_filter(self.config.title_filter)
        chunks, log = self._get_relevant_chunks(list(doc_dict.values()))
        log["docs_after_title_filter"] = list(doc_dict.values())
        log["llm_context"] = None
        log["llm_response_raw"] = None
        log["llm_response"] = self.config.null_value
        self.log.update_document_classification(log)
        if len(chunks) == 0:
            return self.config.null_value
        print("- Querying llm...")
        log["llm_response_raw"], log["llm_context"] = self._query_llm(chunks)
        log["llm_response"] = self._load_response(log["llm_response_raw"])
        self.log.update_document_classification(log)
        return log["llm_response"]

    def extract(self) -> str:
        """
        first checks metadata for variable, then checks documents

        return string response
        """
        metadata_resp = self._extract_from_metadata()
        if metadata_resp != self.config.null_value:
            return metadata_resp
        return self._extract_from_documents()
