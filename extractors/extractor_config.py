"""
This module defines ExtractorConfig, which provides RAG settings for VariableExtractor
"""

from dataclasses import dataclass
from typing import Callable

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
    """
    language_model_prompt: str
    embedding_model_prompt: str
    variable_tag: str
    null_value: str
    content_filter: Callable[[str], bool]
    title_filter: Callable[[str], bool]
    embedding_model: str = "mxbai-embed-large"
    separators: tuple[str] = (
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
        "",)
    chunk_size: int = 700
    overlap: int = 0
    language_model: str = "mistral"
    llm_document_count: int = 7
    document_separator: str = "||"
    llm_context_length: int = 2048
