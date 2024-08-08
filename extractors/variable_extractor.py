"""
Module includes VariableExtractor class
VariableExtractor implements variable extraction from an individual court case
"""
import json
from dataclasses import asdict
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from utils.case_metadata import CaseMetadata
from utils.document import Document
from extractors.extractor_config import ExtractorConfig
from extractors.extractor_log import ExtractorLog

class VariableExtractor:
    """
    VariableExtractor extracts desired variable from case using following process
        first on docket_report then on documents themselves
        filter by title -> chunk documents -> filter by keyword 
        -> rank by semantic similarity -> generate context string -> query llm
    parameters:
        metadata: CaseMetadata for case to extract
        config: ExtractorConfig for desired variable
    public methods:
        from_metadata_path: define extractor using path to case metadata and desired variable
        extract: extract variable using ollama model
        test_context: generates llm prompts and stores logs in self.log
    attributes:
        log: see extractor_log for details
    """
    def __init__(self, metadata: CaseMetadata, config: ExtractorConfig) -> None:
        """
        parameters:
            metadata: CaseMetadata of case to extract
            config: ExtractorConfig for this extractor
        """
        self.metadata = metadata
        self.config = config
        self.log = ExtractorLog(
            config=asdict(self.config),
            metadata=self.metadata.get_metadata_json(fields=("court","title","docket","judges","link")))

    @classmethod
    def from_metadata_path(cls, variable: str, metadata_path: str)->"VariableExtractor":
        """
        Generates variable extractor with correct config to extract variable provided
        Parameters:
            variable: See valid variables in ExtractorConfig.get_config
            metadata_path: path to case metadata.json file
        """
        metadata = CaseMetadata.from_metadata_path(metadata_path)
        return cls(metadata, ExtractorConfig.get_config(variable, metadata))

    def _chunk_documents(self, docs: list[Document]) -> list[str]:
        """
        Given list of documents, chunks documents based on self.config.separators
        """
        text_list = [doc.content_clean for doc in docs]
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

    def _filter_by_semantic_similarity(self, text_list: list[str]) -> list[str]:
        """
        Constructs vector database from provided texts,
        filters documents by semantic similarity to embedding_model_prompt
        """
        if len(text_list) == 0:
            return []
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

    def _get_context_string(self, docs: list[Document]):
        """
        Returns context string after filtering by title,
        chunking, filtering by keyword, and filtering by semantic similarity
        """
        docs_after_title_filter = [doc for doc in docs if self.config.title_filter(doc.title)]
        chunks = self._chunk_documents(docs_after_title_filter)
        chunks_after_keyword_filter = [chunk
                                       for chunk in chunks
                                       if self.config.content_filter(chunk)]
        chunks_after_semantic_filter = self._filter_by_semantic_similarity(chunks_after_keyword_filter)
        context = self.config.document_separator.join(chunks_after_semantic_filter)
        full_prompt = self.config.document_separator.join([self.config.language_model_prompt] + chunks_after_semantic_filter)
        log = {"docs": docs, "docs_after_title_filter": docs_after_title_filter,
                "chunks": chunks, "chunks_after_keyword_filter": chunks_after_keyword_filter,
                "chunks_after_semantic_filter": chunks_after_semantic_filter, "context": context, "full_prompt": full_prompt}
        return (context, log)

    def _query_llm(self, context_string: str) -> tuple[str, str]:
        """
        queries llm based on query and documents given as context
        Parameters:
            context_string
        returns:
            tuple[variable value, full response]
        """
        response = ollama.generate( # pylint: disable=unsubscriptable-object
            model=self.config.language_model,
            prompt=context_string,
            system=self.config.language_model_prompt,
            options={"num_ctx": self.config.llm_context_length},
            format="json",
        )["response"]
        try:
            return (json.loads(response)[self.config.variable_tag], response)
        except: # pylint: disable=bare-except
            return (self.config.null_value, response)

    def _extract_from_metadata(self) -> str:
        """
        queries llm with docket_report content and logs query
        returns variable value
        """
        print("Extracting from metadata...")
        print("- Getting relevant chunks...")
        context, log = self._get_context_string(self.metadata.get_docket_report_as_documents())
        self.log.update_metadata_classification(log)
        if len(context) == 0:
            return self.config.null_value
        print("- Querying llm...")
        var, resp = self._query_llm(context)
        log["variable"] = var
        log["response"] = resp
        self.log.update_metadata_classification(log)
        return var

    def _extract_from_documents(self) -> str:
        """
        queries llm with relevant chunks from documents,
        only uses documents associated with relevant docket_report entry

        returns variable value
        """
        print("Extracting from documents...")
        print("- Getting relevant chunks...")
        context, log = self._get_context_string(self.metadata.get_documents())
        self.log.update_document_classification(log)
        if len(context) == 0:
            return self.config.null_value
        print("- Querying llm...")
        print(repr(context))
        var, resp = self._query_llm(context)
        log["variable"] = var
        log["response"] = resp
        self.log.update_document_classification(log)
        return var

    def extract(self) -> str:
        """
        first checks metadata for variable, then checks documents

        return variable value, stores log in self.log
        """
        metadata_resp = self._extract_from_metadata()
        resp = (metadata_resp
                if metadata_resp != self.config.null_value
                else self._extract_from_documents())
        print(resp)
        return resp

    def test_context(self) -> None:
        """
        Generates context strings for both metadata and document searches
        (these can be accessed using log)
        """
        _, metadata_log = self._get_context_string(self.metadata.get_docket_report_as_documents())
        self.log.update_metadata_classification(metadata_log)
        _, document_log = self._get_context_string(self.metadata.get_documents())
        self.log.update_document_classification(document_log)
