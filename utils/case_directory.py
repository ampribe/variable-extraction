"""
Module includes CaseDirectory path that provides utility functions for handling directory of cases
"""
import os
import numpy as np
import pandas as pd
from utils.case_metadata import CaseMetadata

class CaseDirectory:
    """
    Defines methods for handling directory of case documents

    Methods:
        get_metadata_json: returns list of metadata json objects for each case
        convert_to_text: converts all files in directory to text
        get_proportion_downloaded: returns proportion of total documents downloaded
        get_proportion_downloaded_by_case: returns average documents downloaded per case
    """

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
        court_folders = [
            f.path for f in os.scandir(self.parent_directory) if f.is_dir()
        ]
        case_folders = []
        for court_folder in court_folders:
            case_folders += [f.path for f in os.scandir(court_folder) if f.is_dir]
        metadata_paths = []
        for folder in case_folders:
            path = os.path.join(folder, "metadata.json")
            if os.path.isfile(path):
                metadata_paths.append(path)
        return metadata_paths

    def get_metadata_json(
        self,
        fields: set[str] = (
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
    ) -> list[dict]:
        """
        returns list of metadata.json contents for each case included in directory
        includes above fields and metadata_path
        """
        return [
            CaseMetadata.from_metadata_path(path).get_metadata_json(fields)
            for path in self.metadata_paths
        ]

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
                        os.rename(path, path + ".txt")
                if os.path.isdir(path):
                    convert_directory(path)

        convert_directory(self.parent_directory)

    def get_proportion_downloaded(self) -> float:
        """
        returns proportion of total documents downloaded
        """
        downloaded_count = sum(
            (
                CaseMetadata.from_metadata_path(path).get_downloaded_document_count()
                for path in self.metadata_paths
            )
        )
        total_count = sum(
            (
                CaseMetadata.from_metadata_path(path).get_total_document_count()
                for path in self.metadata_paths
            )
        )
        return np.round(downloaded_count / total_count, 2)

    def get_proportion_downloaded_by_case(self) -> float:
        """
        returns the average proportion of documents downloaded per case
        """
        return np.round(
            np.mean(
                [
                    CaseMetadata.from_metadata_path(path).get_proportion_downloaded()
                    for path in self.metadata_paths
                ]
            ),
            2,
        )

    def categorize_cases(self, title: str) -> None:
        """
        Writes case metadata with additional trial column 
        (whether case went to trial) to categories.csv
        """
        df = pd.DataFrame(self.get_metadata_json())
        df["trial"] = list(CaseMetadata.from_metadata_path(path).categorize_trial()
                       for path in df.metadata_path)
        df.to_csv(title, index=False)
