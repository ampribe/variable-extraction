"""
Module provides DocumentSummarizer class for summarizing case documents
"""
from extractors.variable_extractor import VariableExtractor
from extractors.extractor_config import ExtractorConfig
from utils.case_metadata import CaseMetadata

class DocumentSummarizer(VariableExtractor):
    """
    Provides variable extractor with a system prompt and keyword filter
    for classifying trial type (bench/jury)

    Parameters:
        metadata_path: path to metadata file for case to classify

    Public methods:
        extract: extracts summary from case
    """
    @staticmethod
    def _get_default_config(metadata: CaseMetadata) -> ExtractorConfig:
        language_model_prompt = """
        You are an expert legal analyst. The user message will contain a sequence of documents related to a legal case in the United States.
        Your task is to summarize the events of the case. Describe each document in 10 words or less. You must describe the outcome of the case.
        Documents may contain metadata such as filing date, the name of the person issuing the document, or extra symbols or numbers. These are not important, do not include them in your answer.
        Do not summarize each document. Only summarize the major events during the case. If multiple documents describe redundant information, combine them into one description.
        Respond with a JSON object in the format {"summary": ...} where the summary field includes a string summarizing major case events and the case outcome.
        If the case outcome is unknown, respond with {"summary": "unknown"}
        """
        embedding_model_prompt = "Complaint, orders, outcomes including trial, plea, settlement, judgement, dismissal or dismissed, or verdict"

        def keyword_filter(s):
            result_keywords = [
                "verdict",
                "judgement",
                "opinion",
                "decision",
                "decree",
                "order",
                "ruling",
                "disposition",
                "finding",
                "trial",
                "complaint",
                "order"
            ]
            return any((word in s.lower() for word in result_keywords))

        return ExtractorConfig(
            language_model_prompt=language_model_prompt,
            embedding_model_prompt=embedding_model_prompt,
            variable_tag="summary",
            null_value="unknown",
            content_filter=keyword_filter,
            title_filter=keyword_filter)

