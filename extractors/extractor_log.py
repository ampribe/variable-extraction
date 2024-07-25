"""
Module provides ExtractorLog which records a log of extraction
"""
from dataclasses import dataclass, field

@dataclass
class ExtractorLog:
    """
    Class records extraction metadata (extractor config, metadata of case, output of steps)
    metadata_classification
        docs
        chunks
        chunks_after_keyword_filter
        chunks_after_semantic_filter
        llm_response
    document_classification
        docs
        docs_after_docket_report_filter
        chunks
        chunks_after_keyword_filter
        chunks_after_semantic_filter
        llm_response
    """
    config: dict
    metadata: dict[str, str]
    metadata_classification: dict|None = field(default=None)
    document_classification: dict|None = field(default=None)

    def update_metadata_classification(self, log: dict) -> None:
        self.metadata_classification = log

    def update_document_classification(self, log: dict) -> None:
        self.document_classification = log