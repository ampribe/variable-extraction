"""
Module provides TrialTypeClassifier class for classifying trial type (bench/jury)
"""
from extractors.variable_extractor import VariableExtractor
from extractors.extractor_config import ExtractorConfig
from utils.case_metadata import CaseMetadata

class TrialTypeClassifier(VariableExtractor):
    """
    Provides variable extractor with a system prompt and keyword filter
    for classifying trial type (bench/jury)

    Parameters:
        metadata_path: path to metadata file for case to classify

    Public methods:
        extract: extracts trial type (bench/jury) from case
        summarize: summarizes provided document
    """
    def __init__(
        self,
        metadata_path: str,
    ) -> None:
        """
        Classifies trial type (bench trial or jury trial).
        Response includes reasoning + category (bench trial, jury trial, undetermined)
        Keyword filter requires documents contain outcome keyword
        (verdict, ruling, judgement, etc.)

        """
        language_model_prompt = """
        You are an expert legal analyst. The given documents relate to a legal case in the United States. All descriptions correspond to the same case. Your task is to classify the outcome of this case into one of the following categories:

        jury trial
        bench trial
        undetermined

        Jury trials include documents such as verdict sheets or juror instructions. Bench trials include a decision, a finding, or opinion by the judge. Bench trials do not include references to a jury.
        Some cases are resolved by a plea, a settlement, an agreed judgement, or are dismissed before trial. These cases usually reference a plea agreement, dismissal, settlement, or judgement.
        These cases will not include an opinion, a verdict, or sentencing. 
        If any descriptions identifying the case as a jury trial are found, categorize the case outcome as "jury trial". 
        If any descriptions identifying the case as a bench trial are found, categorize the case outcome as "bench trial". 
        If the descriptions do not identify a trial occuring or the documents are ambiguous, categorize the case outcome as undetermined.


        Respond with a JSON object in the format "{"reasoning": "...", "category": "..."}" 
        If a description identifying the case outcome is found, reasoning should be a string in the form "The event _ during the case supports the case being _." 
        Otherwise, the reasoning should be a string in the form, "The case includes _, which could be found in cases with multiple outcomes. Therefore, the case outcome is undetermined". Do not summarize the case or documents. 
        category should only include one of the following categories as a string: jury trial, bench trial, undetermined.
        """
        embedding_model_prompt = "Case outcome including trial, plea, settlement, judgement, dismissal or dismissed, or verdict"

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
            ]
            return any((word in s.lower() for word in result_keywords))

        config = ExtractorConfig(
            language_model_prompt=language_model_prompt,
            embedding_model_prompt=embedding_model_prompt,
            variable_tag="category",
            null_value="undetermined",
            content_filter=keyword_filter,
            title_filter=keyword_filter)
        super().__init__(
            CaseMetadata.from_metadata_path(metadata_path),
            config
        )
