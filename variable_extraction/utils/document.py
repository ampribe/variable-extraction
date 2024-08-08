"""
Includes Document class for handling individual case documents
"""
from dataclasses import dataclass
from typing import Optional
import os
import regex as re

@dataclass
class Document:
    """
    the Document class stores individual case documents and provides document cleaning methods
    Parameters:
        title: title of document (used as content if path is missing eg. for docket_report entry)
        path: path to document txt file
    """
    title: str
    path: Optional[str] = None

    @property
    def content(self) -> str:
        """
        Loads document content, uses title if no path
        """
        if self.path is None:
            return self.title
        if os.path.isfile(self.path):
            with open(self.path, errors="ignore", encoding="utf-8") as f:
                return f.read()
        return ""

    @property
    def content_clean(self) -> str:
        """
        Returns clean version of content
        by applying each regex substitution defined below
        """
        content = self.content
        subs = [
             {"pattern": r"(\s+\d+){2,}",
              "description": "remove columns of numbers"},
             {"pattern": r"Filed (\d+\/)*\d+",
              "description": "Remove filing date"},
              {"pattern": r"Case( No.| Number| Num| Num.| num.| num| number| no.| no| No)? \w+(-\w+)+",
               "description": "Remove case number, in form Case 0cv-0392BRO-FFM"},
               {"pattern": r"Page\s*ID\s*#?:\s*\d+",
                "description": "Remove page ID in form Page ID #:554"},
                {"pattern": r"Page ?\d* ?of",
                "description": "Remove page number in form Page of (sometimes actual number is removed by above patterns)"},
                {"pattern": r"Document ?\d+",
                "description": "Remove document number"},
                {"pattern": r"(?<=\s{2})\s+",
                "description": "Remove extra whitespace"},
                {"pattern": r"___+",
                 "description": "Remove empty signature lines"}
        ]
        for sub in subs:
            content = re.sub(sub["pattern"], "", content)
        return content
