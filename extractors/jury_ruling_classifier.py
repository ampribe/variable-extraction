"""Module includes JuryRulingClassifier that classifies ruling of jury trial"""
from utils.case_metadata import CaseMetadata
from extractors.variable_extractor import VariableExtractor

class JuryRulingClassifier(VariableExtractor):
    """
    Provides variable extractor with a system prompt and keyword filter
    for classifying jury ruling (ruling for plaintiff or ruling for defendant)

    Parameters:
        metadata_path: path to metadata file for case to classify

    Public methods:
        extract: extracts ruling (plaintiff/defendant) from case
        summarize: summarizes provided document
    """
    def __init__(
        self,
        metadata_path: str,
        embedding_model: str = "mxbai-embed-large",
        separators: list[str]|None = None,
        chunk_size: int = 700,
        overlap: int = 0,
        language_model: str = "mistral",
        llm_document_count: int = 8,
        document_separator: str = "||",
        llm_context_length: int = 2048,
    ) -> None:
        """
        Classifies jury ruling.
        Prompt identifies plaintiffs/defendants by both title and name
        (should classify cases that refer to plaintiff/defendant by name)
        Response includes reasoning + category (plaintiff, defendant, undetermined)
        Keyword filter requires documents contain party keyword
        (plaintiff, defendant, names of plaintiff/defendant)
        and outcome keyword (verdict, ruling, judgement, etc.)

        """
        parties_dict = CaseMetadata.from_metadata_path(metadata_path).get_parties_dict()

        prompt = f"""
        You are an expert legal analyst. You will be given a list of excerpts from legal documents relating to a case in the United States with a decision made by a jury. Documents are separated by ||. All documents correspond to the same case. Classify the outcome of this case into one of the following categories:

        plaintiff
        defendant
        undetermined

        If the jury decided in favor of the plaintiff, classify the outcome as plaintiff. 
        The documents may refer to the plaintiff as "plaintiff" or by name. 
        Here is a list of plaintiff names: {", ".join(parties_dict["plaintiff"][:min(5, len(parties_dict["plaintiff"]))])}
        If the jury decided in favor of the defendant, classify the outcome as defendant.
        The documents may refer to the defendant as "defendant" or by name.
        Here is a list of defendant names: {", ".join(parties_dict["defendant"][:min(5, len(parties_dict["defendant"]))])}
        If the documents provided do not identify the jury verdict or the documents are ambiguous, classify the outcome as undetermined.

        Respond with a JSON object in the format "{{"reasoning": "...", "category": "..."}}"
        If the jury verdict is identified, reasoning should be in the format "According to the documents, _ occurred. This shows that the jury ruled in favor of _ because _."
        If no verdict is identified, reasoning should be in the format "The documents describe _, which does not identify the result of the jury trial."
        Do not summarize the case or documents in reasoning. 
        category should only include one of the following categories: plaintiff, defendant, undetermined.
        """

        party_keywords = [val for ls in parties_dict.values() for val in ls] + [
                "plaintiff",
                "defendant",
            ]
        def has_party(s: str) -> bool:
            return any(map(lambda party: party.lower() in s.lower(), party_keywords))
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
        ]
        def has_result_keywords(s: str) -> bool:
            return any((word.lower() in s.lower() for word in result_keywords)) or "trial" in s
        def content_filter(s: str):
            return has_result_keywords(s) and has_party(s) and "trial" in s
        def title_filter(s: str):
            return has_result_keywords(s)
        

        super().__init__(
            metadata_path,
            prompt,
            "category",
            "undetermined",
            content_filter,
            title_filter,
            embedding_model,
            separators,
            chunk_size,
            overlap,
            language_model,
            llm_document_count,
            document_separator,
            llm_context_length,
        )
