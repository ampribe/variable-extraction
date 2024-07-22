import os
import json
from typing import Callable
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb


class CaseDirectory:
    def __init__(self, parent_directory: str) -> None:
        """
        Provides methods for handling directory of case documents
        Assumes parent directory contains court directories
        Each court directory contains docket directories
        Each docket directory contains metadata.json, docs.json,
        and a directory containing all documents (documents can be in subdirectories)
        """
        self.parent_directory = parent_directory
        self.metadata_paths = self._get_metadata_paths()
    
    def _get_metadata_paths(self) -> list[str]:
        """
        Returns list of paths for each case metadata file
        """
        court_folders = [f.path for f in os.scandir(self.parent_directory) if f.is_dir()]
        case_folders = []
        for court_folder in court_folders:
            case_folders += [f.path for f in os.scandir(court_folder) if f.is_dir]
        metadata_paths = []
        for folder in case_folders:
            path = os.path.join(folder, "metadata.json")
            if os.path.isfile(path):
                metadata_paths.append(path)
        return metadata_paths
    
    def get_metadata_json(self, fields=["court", "title", "docket", "judges",
                                        "judge", "type", "link", "status", "flags",
                                        "nature_of_suit", "cause", "magistrate"]
                                        ) -> list[dict]:
        return [CaseMetadata.from_path(path).get_metadata_json(fields) for path in self.metadata_paths]
    
    def convert_to_text(self) -> None:
        """
        Converts all documents missing a file extension in directory to text files
        """
        def convert_directory(parent: str) -> None:
            for f in os.listdir(parent):
                path = os.path.join(parent, f)
                if os.path.isfile(path):
                    _, ext = os.path.splitext(path)
                    if ext == "":
                        os.rename(path, path+".txt")
                if os.path.isdir(path):
                    convert_directory(path)
        convert_directory(self.parent_directory)

    def get_proportion_downloaded(self) -> float:
        """
        returns proportion of total documents downloaded
        """
        downloaded_count = sum((CaseMetadata.from_path(path).get_downloaded_document_count() for path in self.metadata_paths))
        total_count = sum((CaseMetadata.from_path(path).get_total_document_count() for path in self.metadata_paths))
        return np.round(downloaded_count/total_count, 2)
    
    def get_mean_downloaded_per_case(self) -> float:
        """
        returns the average proportion of documents downloaded per case
        """
        return np.round(np.mean([CaseMetadata.from_path(path).get_proportion_downloaded() for path in self.metadata_paths]), 2)


class CaseMetadata:
    def __init__(self, metadata: str, path: str) -> None:
        self.metadata = json.loads(metadata)
        self.path = path
        self.docs_json = self._get_docs_json()

    @classmethod
    def from_path(cls, path: str) -> "CaseMetadata":
        """
        Initializes CaseMetadata from path to metadata.json
        """
        with open(path) as f:
            return cls(f.read(), path)
    
    def get_metadata_json(self, fields=["court", "title", "docket", "judges",
                                        "judge", "type", "link", "status", "flags",
                                        "nature_of_suit", "cause", "magistrate"]
                                        ) -> dict:
        """
        returns json of standard case metadata (not including docket_report)
        """
        return {field: self.get_info_field(field) for field in fields} | {"metadata_path": self.path}

    def _get_docs_json(self) -> list[dict]:
        """
        returns contents of docs.json file as json
        """
        docs_path = self.path.replace("metadata", "docs")
        with open(docs_path) as f:
            return json.loads(f.read())

    def get_info_field(self, field: str) -> list[str]|str|float:
        """
        Returns field from information section of metadata if it exists,
        otherwise returns np.nan
        Common fields: 
        court, title, docket, judges, judge, type, link, status,
        flags, nature_of_suit, cause, magistrate
        """
        if "info" in self.metadata and field in self.metadata["info"]:
            return self.metadata["info"][field]
        return np.nan
        
    def get_parties_dict(self) -> dict[str, list[str]]:
        """
        returns dictionary in the form 
        {"plaintiff": [plaintiff1, ...], "defendant": [defendant1, ...]}
        """
        parties = {"plaintiff": [], "defendant": []}
        if "parties" in self.metadata:
            for party in self.metadata["parties"]:
                if "type" in party and "name" in party:
                    if "plaintiff" in party["type"].lower():
                        parties["plaintiff"].append(party["name"])
                    if "defendant" in party["type"].lower():
                        parties["defendant"].append(party["name"])
        return parties
    
    def get_docket_report(self) -> pd.DataFrame:
        """
        returns dataframe of docket_report entries with associated metadata
        Adds column for path to document (if downloaded)
        Assumes document names are derived by removing the case link portion 
        of the document link and documents are contained in a folder within
        the same parent directory as metadata.json
        """
        def get_document_path(link_viewer):
            case_link = self.get_info_field("link")
            if type(case_link) == str and type(link_viewer) == str and case_link in link_viewer:
                file = link_viewer.replace(case_link, "")
                file = file[:-1] if file[-1] == "/" else file
                file += ".txt"
                parent = self.path.replace("metadata.json", "")
                paths = [os.path.join(f.path, file) for f in os.scandir(parent) if os.path.isdir(f)]
                for path in paths:
                    if os.path.isfile(path):
                        return path
            return ""
        
        df = pd.json_normalize(self.metadata["docket_report"])
        if "number" in df.columns:
            df.number = pd.to_numeric(df.number)
        if "date" in df.columns:
            df.date = pd.to_datetime(df.date)
        if "entry_date" in df.columns:
            df.entry_date = pd.to_datetime(df.entry_date)
        if "contents" in df.columns:
            df.contents = df.contents.apply(lambda html: BeautifulSoup(html, features="html.parser").text)
        if "link_viewer" in df.columns:
            df["document_path"] = df.link_viewer.apply(get_document_path)
        return df

    def get_docket_report_content(self) -> list[str]:
        """
        returns list of docket_report entries 
        (ignores associate metadata like date, index, etc)
        """
        entries = []
        if "docket_report" in self.metadata:
            for docket in self.metadata["docket_report"]:
                if "contents" in docket:
                    entries.append(BeautifulSoup(docket["contents"], features="html.parser").text)
        return entries
    
    def get_documents(self) -> list[str]:
        """
        returns list of documents found in 
        subdirectory of metadata parent folder
        (checks all subdirectories within parent folder)
        """
        def search_subdirectory(path: str) -> list[str]:
            docs = []
            for entry in os.scandir(path):
                if os.path.isdir(entry):
                    docs += search_subdirectory(entry.path)
                else:
                    with open(entry.path, errors='ignore') as f:
                        docs.append(f.read())
            return docs
        parent_directory = os.path.dirname(self.path)
        documents = []
        for f in os.scandir(parent_directory):
            if os.path.isdir(f):
                documents += search_subdirectory(f.path)
        return documents
    
    def get_total_document_count(self) -> int:
        return len(self.get_docket_report_content())
    
    def get_downloaded_document_count(self) -> int:
        return len(self.get_documents())
    
    def get_proportion_downloaded(self) -> float:
        """
        returns proportion of documents downloaded
        assumes each entry in docket_report is a document
        and each downloaded file is a separate document
        """
        return self.get_downloaded_document_count()/self.get_total_document_count()
    
class VariableExtractor:
    def __init__(self,
                 metadata_path: str,
                 prompt: str,
                 variable_tag: str,
                 null_value: str,
                 filter_function: Callable[[str], bool],
                 embedding_model: str = "mxbai-embed-large",
                 separators: list[str] = ["\n\n", "\n", ". ", "!", "?", ".", ";",":", ",", " ", ""],
                 chunk_size: int = 700,
                 overlap: int = 0,
                 language_model: str = "mistral",
                 llm_document_count: int = 9,
                 document_separator: str = "||",
                 llm_context_length: int = 2048) -> None:
        """
        parameters:
            metadata_path: path to docket metadata (assumes documents stored in the same parent directory)
            prompt: instructions for llm to evaluate provided documents, includes desired json formatting (eg. {reasoning: ..., variable: ...} for chain of thought)
            variable_tag: name of variable of interest in json returned by llm eg. "variable"
            null_value: value for variable in json that indicates extraction failed (used to query documents if metadata query fails) eg. "unknown"
            filter_function: function used to filter text chunks, must take string and return boolean
            embedding_model: model used to generate prompt and text embedding, models available: https://ollama.com/blog/embedding-models
            separators: separators used to chunk text in order of preference
            chunk_size: maximum size of each text chunk (in characters)
            overlap: overlap between chunks (in characters)
            language_model: model used to evaluate prompt, models available: https://ollama.com/library
            llm_document_count: number of documents to pass to llm
            document_separator: string used to separate documents passed to llm
            llm_context_length: length of llm context window (in tokens), must cover both system prompt and provided documents
        note metadata stored as CaseMetadata
        """
        self.metadata = CaseMetadata.from_path(metadata_path)
        self.prompt = prompt
        self.variable_tag = variable_tag
        self.null_value = null_value
        self.filter_function = filter_function
        self.embedding_model = embedding_model
        self.separators = separators
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.language_model = language_model
        self.llm_document_count = llm_document_count
        self.document_separator = document_separator
        self.llm_context_length = llm_context_length

    def get_docket_report_content(self) -> list[str]:
        """
        returns list of docket_report entries in metadata 
        (ignores associated metadata like date entered, index, etc.)
        """
        return self.metadata.get_docket_report_content()
    
    def get_documents(self) -> list[str]:
        """
        returns list of documents found in 
        subdirectories of metadata parent folder
        """
        return self.metadata.get_documents()

    def chunk_text_list(self, text_list: list[str]) -> list[str]:
        """
        chunks list of texts using parameters given to constructor
        """
        text_splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
        )
        components = []
        for text in text_list:
            components += text_splitter.split_text(text)
        return components
    
    def filter_by_keyword_function(self, text_list: list[str]) -> list[str]:
        """
        filters texts by provided keyword filtering function
        """
        return [text for text in text_list if self.filter_function(text)]

    def filter_by_semantic_similarity(self, text_list: list[str]) -> list[str]:
        """
        Adds each document to vector database and returns documents most similar
        to given prompt
        """
        client = chromadb.Client()
        try:
            collection = client.create_collection("case")
        except:
            client.delete_collection("case")
            collection = client.create_collection("case")
        for i, text in enumerate(text_list):
            if len(text) > 0:
                embedding = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text)["embedding"]
                collection.add(
                    ids=[str(i)],
                    embeddings=[embedding],
                    documents=[text]
                )
        prompt_embedding = ollama.embeddings(
            model=self.embedding_model,
            prompt=self.prompt)["embedding"]
        return collection.query(
            query_embeddings=[prompt_embedding],
            n_results=self.llm_document_count
        )["documents"][0]
    
    def get_relevant_chunks(self, docs: list[str]) -> list[str]:
        """
        Applies chunking, then keyword filter, then semantic similarity filter to given document list
        """
        return self.filter_by_semantic_similarity(self.filter_by_keyword_function(self.chunk_text_list(docs)))
    
    def query_llm(self, context_documents: list[str]) -> tuple[dict[str, str], str]:
        """
        queries llm based on query and documents given as context
        query (from parameters) given as system prompt, documents given as prompt
        returns tuple of response (after converting to json) and context provided from documents
        """
        context = self.document_separator.join(context_documents)
        response = ollama.generate(
            model=self.language_model,
            prompt=context,
            system=self.prompt,
            options={"num_ctx": self.llm_context_length},
            format="json"
        )["response"]
        return (json.loads(response), context)
    
    def extract_from_metadata(self) -> tuple[dict[str, str], str]:
        """
        queries llm with docket_report content
        returns tuple of response and documents provided as context
        """
        relevant_chunks = self.get_relevant_chunks(self.get_docket_report_content())
        if len(relevant_chunks) == 0:
            return ({self.variable_tag: self.null_value}, "No relevant docket_report entries")
        return self.query_llm(relevant_chunks)
        
    def extract_from_documents(self) -> tuple[dict[str, str], str]:
        """
        queries llm with relevant chunks from documents
        returns tuple of response and documents provided as context 
        """
        relevant_chunks = self.get_relevant_chunks(self.get_documents())
        if len(relevant_chunks) == 0:
            ({self.variable_tag: self.null_value}, "No relevant documents")
        return self.query_llm(relevant_chunks)
    
    def extract(self) -> tuple[dict[str, str], dict[str, str]]:
        """
        first checks metadata for variable, then checks documents

        return tuple of response and log
        """
        log = {}
        metadata_resp, metadata_context = self.extract_from_metadata()
        log["metadata_response"] = metadata_resp
        log["metadata_context"] = metadata_context
        if metadata_resp[self.variable_tag] != self.null_value:
            return (metadata_resp, log)
        document_resp, document_context = self.extract_from_documents()
        log["document_response"] = document_resp
        log["document_context"] = document_context
        return (document_resp, log)
    
    def summarize(self, document) -> str:
        """
        summarizes legal document
        """
        system_prompt = """
        You are an expert legal analyst. The user message will contain a sequence of documents related to a legal case in the United States.
        Your task is to summarize the events of the case. Describe each document in 10 words or less.
        Documents may contain metadata such as filing date or the name of the person issuing the document. These are not important, do not include them in your answer.
        Documents may include extra symbols or numbers that do not describe what the document is. Do not include these in your answer.
        Do not summarize each document. Only summarize the major events during the case. If multiple documents describe redundant information, combine them into one description.
        """
        return ollama.generate(
            model=self.language_model,
            prompt=document,
            system=system_prompt,
            options={"num_ctx": self.llm_context_length},
        )["response"]


class JuryRulingClassifier(VariableExtractor):
    def __init__(self,
                 metadata_path: str,
                 embedding_model: str = "mxbai-embed-large",
                 separators: list[str] = ["\n\n", "\n", ". ", "!", "?", ".", ";", ":", ",", " ", ""],
                 chunk_size: int = 700,
                 overlap: int = 0,
                 language_model: str = "mistral",
                 llm_document_count: int = 9,
                 document_separator: str = "||",
                 llm_context_length: int = 2048) -> None:
        """
        Classifies jury ruling.
        Prompt identifies plaintiffs/defendants by both title and name
        (should classify cases that refer to plaintiff/defendant by name)
        Response includes reasoning + category (plaintiff, defendant, undetermined)
        Keyword filter requires documents contain party keyword
        (plaintiff, defendant, names of plaintiff/defendant)
        and outcome keyword (verdict, ruling, judgement, etc.)

        """
        parties_dict = CaseMetadata.from_path(metadata_path).get_parties_dict()
        
        prompt = f"""
        You are an expert legal analyst. You will be given a list of excerpts from legal documents relating to a case in the United States with a decision made by a jury. Documents are separated by ||. All documents correspond to the same case. Classify the outcome of this case into one of the following categories:

        plaintiff
        defendant
        undetermined

        If the jury decided in favor of the plaintiff, classify the outcome as plaintiff. The documents may refer to the plaintiff as "plaintiff" or by name. Here is a list of plaintiff names: {", ".join(parties_dict["plaintiff"])}
        If the jury decided in favor of the defendant, classify the outcome as defendant. The documents may refer to the defendant as "defendant" or by name. Here is a list of defendant names: {", ".join(parties_dict["defendant"])}
        If the documents provided do not identify the jury verdict or the documents are ambiguous, classify the outcome as undetermined.

        Respond with a JSON object in the format "{{"reasoning": "...", "category": "..."}}"
        If the jury verdict is identified, reasoning should be in the format "According to the documents, _ occurred. This shows that the jury ruled in favor of _ because _."
        If no verdict is identified, reasoning should be in the format "The documents describe _, which does not identify the result of the jury trial."
        Do not summarize the case or documents in reasoning. 
        category should only include one of the following categories: plaintiff, defendant, undetermined.
        """

        def keyword_filter(s):
            party_keywords = [val for ls in parties_dict.values() for val in ls] + ["plaintiff", "defendant"]
            has_party = lambda s: any(map(lambda party: party.lower() in s.lower(), party_keywords))
            result_keywords = ["verdict", "judgement", "opinion", "decision", "decree", "order", "ruling", "disposition", "finding", "trial"]
            has_result_keywords = lambda s: any([word in s.lower() for word in result_keywords])
            return has_result_keywords(s) and has_party(s)
        

        super().__init__(metadata_path,
                         prompt,
                         "category",
                         "undetermined",
                         keyword_filter,
                         embedding_model,
                         separators,
                         chunk_size,
                         overlap,
                         language_model,
                         llm_document_count,
                         document_separator,
                         llm_context_length)


class BenchRulingClassifier(VariableExtractor):
    def __init__(self,
                 metadata_path: str,
                 embedding_model: str = "mxbai-embed-large",
                 separators: list[str] = ["\n\n", "\n", ". ", "!", "?", ".", ";", ":", ",", " ", ""],
                 chunk_size: int = 700,
                 overlap: int = 0,
                 language_model: str = "mistral",
                 llm_document_count: int = 9,
                 document_separator: str = "||",
                 llm_context_length: int = 2048) -> None:
        """
        Classifies bench trial ruling.
        Prompt identifies plaintiffs/defendants by both title and name
        (should classify cases that refer to plaintiff/defendant by name)
        Response includes reasoning + category (plaintiff, defendant, undetermined)
        Keyword filter requires documents contain party keyword
        (plaintiff, defendant, names of plaintiff/defendant)
        and outcome keyword (verdict, ruling, judgement, etc.)

        """
        parties_dict = CaseMetadata.from_path(metadata_path).get_parties_dict()
        
        prompt = f"""
        You are an expert legal analyst. You will be given a list of excerpts from legal documents relating to a case in the United States with a decision made by a judge. Documents are separated by ||. All documents correspond to the same case. Classify the outcome of this case into one of the following categories:

        plaintiff
        defendant
        undetermined

        If the judge decided in favor of the plaintiff, classify the outcome as plaintiff. The documents may refer to the plaintiff as "plaintiff" or by name. Here is a list of plaintiff names: {", ".join(parties_dict["plaintiff"])}
        If the judge decided in favor of the defendant, classify the outcome as defendant. The documents may refer to the defendant as "defendant" or by name. Here is a list of defendant names: {", ".join(parties_dict["defendant"])}
        If the documents provided do not identify the judge's ruling or the documents are ambiguous, classify the outcome as undetermined.

        Respond with a JSON object in the format "{{"reasoning": "...", "category": "..."}}"
        If a ruling is identified, reasoning should be in the format "According to the documents, _ occurred. This shows that the judge ruled in favor of _ because _."
        If no ruling is identified, reasoning should be in the format "The documents describe _, which does not identify the result of the trial."
        Do not summarize the case or documents in reasoning. 
        category should only include one of the following categories: plaintiff, defendant, undetermined.
        """

        def keyword_filter(s):
            party_keywords = [val for ls in parties_dict.values() for val in ls] + ["plaintiff", "defendant"]
            has_party = lambda s: any(map(lambda party: party.lower() in s.lower(), party_keywords))
            result_keywords = ["verdict", "judgement", "opinion", "decision", "decree", "order", "ruling", "disposition", "finding", "trial"]
            has_result_keywords = lambda s: any([word in s.lower() for word in result_keywords])
            return has_result_keywords(s) and has_party(s)
        

        super().__init__(metadata_path,
                         prompt,
                         "category",
                         "undetermined",
                         keyword_filter,
                         embedding_model,
                         separators,
                         chunk_size,
                         overlap,
                         language_model,
                         llm_document_count,
                         document_separator,
                         llm_context_length)


class TrialTypeClassifier(VariableExtractor):
    def __init__(self,
                 metadata_path: str,
                 embedding_model: str = "mxbai-embed-large",
                 separators: list[str] = ["\n\n", "\n", ". ", "!", "?", ".", ";", ":", ",", " ", ""],
                 chunk_size: int = 700,
                 overlap: int = 0,
                 language_model: str = "mistral",
                 llm_document_count: int = 9,
                 document_separator: str = "||",
                 llm_context_length: int = 2048) -> None:
        """
        Classifies trial type (bench trial or jury trial).
        Response includes reasoning + category (bench trial, jury trial, undetermined)
        Keyword filter requires documents contain outcome keyword
        (verdict, ruling, judgement, etc.)

        """
        prompt = """
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

        def keyword_filter(s):
            result_keywords = ["verdict", "judgement", "opinion", "decision", "decree", "order", "ruling", "disposition", "finding", "trial"]
            return any([word in s.lower() for word in result_keywords])        

        super().__init__(metadata_path,
                         prompt,
                         "category",
                         "undetermined",
                         keyword_filter,
                         embedding_model,
                         separators,
                         chunk_size,
                         overlap,
                         language_model,
                         llm_document_count,
                         document_separator,
                         llm_context_length)
