"""
Module includes CaseMetadata class that provides methods for handling case documents and metadata
"""
import os
import json
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


class CaseMetadata:
    """
    Provides methods for handling individual cases and case metadata
    Methods:
        get_metadata_json: returns dictionary of standard metadata fields
        get_info_field: returns provided field from info section of metadata
        get_parties_dict: returns dictionary of plaintiffs/defendants
        get_docket_report: returns metadata docket_report with paths to downloaded entries
        get_docket_report_contents: returns list of docket_report entries
        get_documents: returns list of all case document texts
        get_total_document_count: returns number of documents included in docket_report
        get_downloaded_document_count: returns number of docket_report documents downloaded
        get_proportion_downloaded: returns proportion of documents downloaded
    """
    def __init__(self, metadata: str, path: str) -> None:
        self.metadata = json.loads(metadata)
        self.path = path
        self.docs_json = self._get_docs_json()

    @classmethod
    def from_metadata_path(cls, path: str) -> "CaseMetadata":
        """
        Initializes CaseMetadata from path to metadata.json
        """
        with open(path, encoding="utf-8") as f:
            return cls(f.read(), path)

    def get_metadata_json(
        self,
        fields: set[str]=(
            "court",
            "title",
            "docket",
            "judges",
            "judge",
            "type",
            "link",
            "status",
            "flags",
            "nature_of_suit",
            "cause",
            "magistrate",
        ),
    ) -> dict:
        """
        returns json of standard case metadata (not including docket_report)
        """
        return {field: self.get_info_field(field) for field in fields} | {
            "metadata_path": self.path
        }

    def _get_docs_json(self) -> list[dict]:
        """
        returns contents of docs.json file as json
        """
        docs_path = self.path.replace("metadata", "docs")
        with open(docs_path, encoding="utf-8") as f:
            return json.loads(f.read())

    def get_info_field(self, field: str) -> list[str] | str | float:
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
            if (
                isinstance(case_link, str)
                and isinstance(link_viewer, str)
                and case_link in link_viewer
            ):
                file = link_viewer.replace(case_link, "")
                file = file[:-1] if file[-1] == "/" else file
                file += ".txt"
                parent = self.path.replace("metadata.json", "")
                paths = [
                    os.path.join(f.path, file)
                    for f in os.scandir(parent)
                    if os.path.isdir(f)
                ]
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
            df.contents = df.contents.apply(
                lambda html: BeautifulSoup(html, features="html.parser").text
            )
        if "link_viewer" in df.columns:
            df["document_path"] = df.link_viewer.apply(get_document_path)
        return df

    def get_docket_report_contents(self) -> list[str]:
        """
        returns list of docket_report entries
        (ignores associate metadata like date, index, etc)
        """
        return self.get_docket_report().contents.tolist()

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
                    with open(entry.path, errors="ignore", encoding="utf-8") as f:
                        docs.append(f.read())
            return docs

        parent_directory = os.path.dirname(self.path)
        documents = []
        for f in os.scandir(parent_directory):
            if os.path.isdir(f):
                documents += search_subdirectory(f.path)
        return documents

    def get_total_document_count(self) -> int:
        """
        returns total documents in docket_report
        """
        return len(self.get_docket_report_contents())

    def get_downloaded_document_count(self) -> int:
        """
        returns number of documetns downloaded
        """
        return len(self.get_documents())

    def get_proportion_downloaded(self) -> float:
        """
        returns proportion of documents downloaded
        assumes each entry in docket_report is a document
        and each downloaded file is a separate document
        """
        return self.get_downloaded_document_count() / self.get_total_document_count()

