from dataclasses import dataclass
from typing import Optional
import os
import regex as re

@dataclass
class Document:
    title: str
    path: Optional[str] = None

    @property
    def content(self) -> str:
        if self.path is None:
            return self.title
        if os.path.isfile(self.path):
            with open(self.path, errors="ignore", encoding="utf-8") as f:
                return f.read()
        return ""

    @property
    def content_clean(self) -> str:
        content = self.content
        subs = [
            {"pattern": r"(\d[\s|\W])+",
             "description": "Remove extraneous numbers"},
             {"pattern": r"(\d\n)+",
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
