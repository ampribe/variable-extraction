"""
Module provides ExtractorLog which records a log of extraction
"""
from dataclasses import dataclass, field

@dataclass
class ExtractorLog:
    """
    ExtractorLog records extraction metadata
    Fields:
        config: dictionary of ExtractorConfig fields
            language_model_prompt
            embedding_model_prompt
            variable_tag
            null_value
            content_filter
            title_filter
            embedding_model
            separators
            chunk_size
            overlap
            language_model
            llm_document_count
            document_separator
            llm_context_length
        metadata: dictionary of case metadata
            court
            title
            docket
            judges
            link
            metadata_path
        metadata_classification
            docs: list of all docket_report entries
            docs_after_title_filter
            chunks
            chunks_after_keyword_filter
            chunks_after_semantic_filter
            context
            full_prompt
        metadata_classification
            docs: list of all downloaded documents
            docs_after_title_filter
            chunks
            chunks_after_keyword_filter
            chunks_after_semantic_filter
            context
            full_prompt
    """
    config: dict
    metadata: dict[str, str]
    metadata_classification: dict|None = field(default=None)
    document_classification: dict|None = field(default=None)

    def update_metadata_classification(self, log: dict) -> None:
        """
        sets metadata_classification field
        """
        self.metadata_classification = log

    def update_document_classification(self, log: dict) -> None:
        """
        sets document_classification field
        """
        self.document_classification = log