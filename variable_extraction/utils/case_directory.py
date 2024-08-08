"""
Module includes CaseDirectory path that provides utility functions for handling directory of cases
"""
import os
import numpy as np
import pandas as pd
import dill
from .case_metadata import CaseMetadata
from ..extractors.extractor_log import ExtractorLog
from ..extractors.variable_extractor import VariableExtractor

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
        fields: set[str]|None = None,
    ) -> list[dict]:
        """
        returns list of metadata.json contents for each case included in directory
        includes above fields and metadata_path
        """
        if fields is None:
            fields = (
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
        )
        return [
            CaseMetadata.from_metadata_path(path).get_metadata_json(fields)
            for path in self.metadata_paths
        ]

    def get_metadata_df(self, fields: set[str]|None = None) -> pd.DataFrame:
        """
        generates dataframe of provided metadata fields for all cases in directory
        """
        return pd.DataFrame(self.get_metadata_json(fields))

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

    def write_metadata(self, title: str) -> None:
        """
        Writes case metadata with additional trial and trial type column s
        """
        df = self.get_metadata_df()
        metadata = [CaseMetadata.from_metadata_path(path) for path in df.metadata_path]

        df["trial"] = [m.categorize_trial() for m in metadata]
        df["trial_type"] = ([m.categorize_trial_type() if df.iloc[i].trial
                             else "unknown" for i, m in enumerate(metadata)])
        df.to_csv(title, index=False)

    def categorize_outcomes(self, metadata_title: str, log_title: str, result_title: str) -> None:
        """
        Categorizes outcomes of cases with trials
        Parameters:
            metadata_title: filename of metadata file already generated using write_metadata
            log_title: title of log file
            result_title: title of results file
        """
        d = os.path.dirname(os.path.abspath(__file__))
        metadata = pd.read_csv(metadata_title)
        metadata["result"] = ""
        logs = []
        tot = len(metadata[metadata.trial_type != "unknown"])
        for i, (j, row) in enumerate(metadata[metadata.trial_type != "unknown"].iterrows()):
            print(f"\n\n\nClassifying case {i+1} of {tot}")
            if row.trial_type == "bench":
                classifier = VariableExtractor.from_metadata_path("bench_ruling", f"{d}/../{row.metadata_path}")
            else:
                classifier = VariableExtractor.from_metadata_path("jury_ruling", f"{d}/../{row.metadata_path}")
            metadata.loc[j, "result"], log = classifier.extract()
            logs.append(log)

        metadata.to_csv(result_title, index=False)
        with open(log_title, "wb") as f:
            dill.dump(logs, f)

    @staticmethod
    def categorize_from_metadata_path(path: str) -> tuple[str, ExtractorLog|None]:
        """
        Categorizes case outcome as plaintiff, defendant, or unknown
        returns log along with result
        parameters:
            path: path to metadata file generated using write_metadata
        """
        metadata = CaseMetadata.from_metadata_path(path)
        if metadata.categorize_trial_type() == "jury":
            classifier = VariableExtractor.from_metadata_path("jury_ruling", path)
            return (classifier.extract(), classifier.log)
        if metadata.categorize_trial_type() == "bench":
            classifier = VariableExtractor.from_metadata_path("bench_ruling", path)
            return (classifier.extract(), classifier.log)
        return ("unknown", None)

    @staticmethod
    def get_prompt_logs(metadata_path: str) -> list[ExtractorLog]:
        """
        Generate trial outcome classification prompt and record logs
        parameters:
            metadata_path: path to metadata file generated using write_metadata
        """
        d = os.path.dirname(os.path.abspath(__file__))
        metadata = pd.read_csv(metadata_path)
        logs = []
        trial_cases = metadata[(metadata.trial_type == "jury") | (metadata.trial_type == "bench")].reset_index(drop=True)
        for i, row in enumerate(trial_cases.iterrows()):
            print(f"Generating prompt for case {i+1} of {len(trial_cases)}")
            if row.trial_type == "bench":
                classifier = VariableExtractor.from_metadata_path("bench_ruling", f"{d}/../{row.metadata_path}")
            else:
                classifier = VariableExtractor.from_metadata_path("jury_ruling", f"{d}/../{row.metadata_path}")
            classifier.test_context()
            logs.append(classifier.log)
        return logs

    @staticmethod
    def get_prompts(metadata_path: str) -> pd.DataFrame:
        """
        Generate trial outcome classification prompts and returns as dataframe
        parameters:
            metadata_path: path to metadata file generated using write_metadata
        """
        d = os.path.dirname(os.path.abspath(__file__))
        metadata = pd.read_csv(metadata_path)
        trial_cases = metadata[(metadata.trial_type == "jury") | (metadata.trial_type == "bench")].reset_index(drop=True)
        for i, row in enumerate(trial_cases.iterrows()):
            print(f"Generating prompt for case {i+1} of {len(trial_cases)}")
            if row.trial_type == "bench":
                classifier = VariableExtractor.from_metadata_path("bench_ruling", f"{d}/../{row.metadata_path}")
            else:
                classifier = VariableExtractor.from_metadata_path("jury_ruling", f"{d}/../{row.metadata_path}")
            classifier.test_context()
            trial_cases.loc[i, "metadata_prompt"] = classifier.log.metadata_classification["full_prompt"]
            trial_cases.loc[i, "document_prompt"] = classifier.log.document_classification["full_prompt"]
        return trial_cases
