"""
This module defines ExtractorConfig, which provides RAG settings for VariableExtractor
"""
from typing import Callable
from dataclasses import dataclass
from utils.case_metadata import CaseMetadata

@dataclass
class ExtractorConfig:
    """
    Configuration for VariableExtractor

    Parameters:
        language_model_prompt: instructions for llm to evaluate provided documents,
            should include desired json formatting (eg. {reasoning: ..., variable: ...})
        embedding_model_prompt: prompt to provide embedding model when finding relevant documents,
            separate from language_model_prompt because language_model_prompt may contain irrelevant 
            context
        variable_tag: name of variable of interest in json returned by llm eg. "variable", "damages"
        null_value: value for variable in json that indicates extraction failed
            (indicator to query documents if metadata query fails) eg. "unknown"
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
    """
    language_model_prompt: str
    embedding_model_prompt: str
    variable_tag: str
    null_value: str
    content_filter: Callable[[str], bool]
    title_filter: Callable[[str], bool]
    embedding_model: str = "mxbai-embed-large"
    separators: tuple[str] = (
        ".\n\n",
        ".\n",
        ". \n",
        "!",
        "?",
        ".",
        "\n",
        ";",
        ":",
        ",",
        " ",
        "",)
    chunk_size: int = 700
    overlap: int = 0
    language_model: str = "llama3.1"
    llm_document_count: int = 6
    document_separator: str = "\n\n---NEW DOCUMENT---\n"
    llm_context_length: int = 2048

    @classmethod
    def get_config(cls, variable: str, metadata: CaseMetadata) -> "ExtractorConfig":
        """
        Given the variable of interest, defines the proper extractor config
        Possible variables:
            "bench_ruling"
            "jury_ruling"
            "damages"
            "trial_type"
            "summary"
        """
        parameters = {
            "bench_ruling": get_bench_ruling_config,
            "jury_ruling": get_jury_ruling_config,
            "damages": get_damages_config,
            "trial_type": get_trial_type_config,
            "summary": get_summary_config,
        }
        if variable in parameters:
            return parameters[variable](metadata)
        raise Exception(f"ExtractorConfig for {variable} not defined")

def get_bench_ruling_config(metadata: CaseMetadata) -> "ExtractorConfig":
    """
    defines configuration file for extracting ruling of bench trial
    """
    parties_dict = metadata.get_parties_dict()

    language_model_prompt = f"""You are an expert legal analyst. You will be given a list of excerpts from legal documents relating to a case in the United States with a decision made by a judge.
Documents are separated by "---NEW DOCUMENT---". All documents correspond to the same case. Classify the outcome of this case into one of the following categories:

plaintiff
defendant
undetermined

If the judge decided in favor of the plaintiff, classify the outcome as plaintiff.
The documents may refer to the plaintiff as "plaintiff" or by name.
Here is a list of plaintiff names: {", ".join(parties_dict["plaintiff"][:min(5, len(parties_dict["plaintiff"]))])}
If the judge decided in favor of the defendant, classify the outcome as defendant.
The documents may refer to the defendant as "defendant" or by name.
Here is a list of defendant names: {", ".join(parties_dict["defendant"][:min(5, len(parties_dict["defendant"]))])}
If the documents provided do not identify the judge's ruling or the documents are ambiguous, classify the outcome as undetermined.

Respond with a JSON object in the format "{{"reasoning": "...", "category": "..."}}"
If a ruling is identified, reasoning should be in the format "According to the documents, _ occurred. This shows that the judge ruled in favor of _ because _."
If no ruling is identified, reasoning should be in the format "The documents describe _, which does not identify the result of the trial."
Do not summarize the case or documents in reasoning. 
category should only include one of the following categories: plaintiff, defendant, undetermined.
        """
    embedding_model_prompt = "Ruling for plaintiff or defendant in judgement, opinion, decision, or verdict."

    party_keywords = [val for ls in parties_dict.values() for val in ls] + [
            "plaintiff",
            "defendant",
        ]
    def has_party(s: str) -> bool:
        return any(map(lambda party: party.lower() in s.lower(), party_keywords))
    result_keywords = [
        "verdict",
        "judgement",
        "judgment",
        "opinion",
        "decision",
        "decree",
        "order",
        "ruling",
        "disposition",
        "finding",
        "trial"
    ]
    def has_result_keywords(s: str) -> bool:
        return any((word.lower() in s.lower() for word in result_keywords))
    def content_filter(s: str):
        return has_result_keywords(s) and has_party(s)
    def title_filter(s: str):
        return has_result_keywords(s)

    return ExtractorConfig(
        language_model_prompt=language_model_prompt,
        embedding_model_prompt=embedding_model_prompt,
        variable_tag="category",
        null_value="undetermined",
        content_filter=content_filter,
        title_filter=title_filter,
        document_separator="\n\n\n---NEW DOCUMENT---\n")

def get_jury_ruling_config(metadata: CaseMetadata) -> ExtractorConfig:
    """
    defines configuration file for extracting ruling of jury trial
    """
    parties_dict = metadata.get_parties_dict()

    language_model_prompt = f"""You are an expert legal analyst. You will be given a list of excerpts from legal documents relating to a case in the United States with a decision made by a jury.
Documents are separated by "---NEW DOCUMENT---". All documents correspond to the same case. Classify the outcome of this case into one of the following categories:

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

A party proposing a verdict or verdict sheet does not indicate that the jury will rule in favor of that party. Do not use proposed forms in your classification. 

Respond with a JSON object in the format "{{"reasoning": "...", "category": "..."}}"
If the jury verdict is identified, reasoning should be in the format "According to the documents, _ occurred. This shows that the jury ruled in favor of _ because _."
If no verdict is identified, reasoning should be in the format "The documents describe _, which does not identify the result of the jury trial."
Do not summarize the case or documents in reasoning. 
category should only include one of the following categories: plaintiff, defendant, undetermined.
"""
    embedding_model_prompt = "Jury verdict for plaintiff or defendant in judgement, opinion, decision, or verdict."

    party_keywords = [val
                        for ls in parties_dict.values()
                        for val in ls] + [
            "plaintiff",
            "defendant",
        ]
    def has_party(s: str) -> bool:
        return any(map(lambda party: party.lower() in s.lower(), party_keywords))
    result_keywords = [
        "verdict",
        "judgement",
        "judgment",
        "opinion",
        "decision",
        "decree",
        "order",
        "ruling",
        "disposition",
        "finding",
        "trial"
    ]
    def has_result_keywords(s: str) -> bool:
        return any((word.lower() in s.lower() for word in result_keywords))
    def content_filter(s: str):
        return has_result_keywords(s) and has_party(s) and "proposed" not in s.lower()
    def title_filter(s: str):
        return has_result_keywords(s)

    return ExtractorConfig(
        language_model_prompt=language_model_prompt,
        embedding_model_prompt=embedding_model_prompt,
        content_filter=content_filter,
        title_filter=title_filter,
        variable_tag="category",
        null_value="undetermined",
        document_separator="\n\n\n---NEW DOCUMENT---\n")

def get_damages_config(metadata: CaseMetadata) -> ExtractorConfig:
    """
    defines configuration file for extracting case damages
    """
    parties_dict = metadata.get_parties_dict()
    language_model_prompt = f"""You are an expert legal analyst. You will be given a list of excerpts from legal documents relating to a case in the United States.
Documents are separated by "---NEW DOCUMENT---". All documents correspond to the same case.
Your task is to identify the damages that the jury or the judge decide that the defendant must pay the plaintiff. 

The documents may refer to the defendant as "defendant" or by name.
Here is a list of defendant names: {", ".join(parties_dict["defendant"][:min(5, len(parties_dict["defendant"]))])}
The documents may refer to the plaintiff as "plaintiff" or by name. 
Here is a list of plaintiff names: {", ".join(parties_dict["plaintiff"][:min(5, len(parties_dict["plaintiff"]))])}

Respond with a JSON object in the format "{{"reasoning": "...", "damages": "..."}}"
If the damages are identified, reasoning should be in the format "The documents say _. This shows that the defendant was required to pay the plaintiff _."
If the damages are not identified, reasoning should be in the format "The documents provided describe _, which does not identify the damages". 
If the damages are identified, "damages" should contain the dollar amount of the damages. If the damages are not identified, "damages" should be "unknown".
"""
    embedding_model_prompt = "dollar or number of damages in judgement, verdict, decision, or opinion"

    def keyword_filter(s):
        result_keywords = [
            "verdict",
            "judgement",
            "opinion",
            "decision",
            "award",
            "order",
            "ruling",
            "damages",
            "pay",
            "trial",
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

def get_trial_type_config(_: CaseMetadata) -> ExtractorConfig:
    """
    defines configuration for extracting trial type (jury, bench, neither)
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

    return ExtractorConfig(
        language_model_prompt=language_model_prompt,
        embedding_model_prompt=embedding_model_prompt,
        variable_tag="category",
        null_value="undetermined",
        content_filter=keyword_filter,
        title_filter=keyword_filter)

def get_summary_config(_: CaseMetadata) -> ExtractorConfig:
    """
    defines configuration for generating summary of case
    """
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
