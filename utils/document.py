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
            {"pattern": "(\d[\s|\W])+",
             "description": "Remove extraneous numbers"},
             {"pattern": "Filed (\d+\/)*\d+",
              "description": "Remove filing date"}
        ]
        for sub in subs:
            content = re.sub(sub["pattern"], "", content)
        return content
