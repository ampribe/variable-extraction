"""
Tests outcome classification of federal cases that contain document specifying outcome
Requires federal data to be saved to data/100_random_fed
Test results are saved to results.csv with llm classification as "result"
and correct classification as "category"

Logs for each classification are saved as logs.pkl 
Logs can be loaded using "logs = dill.load(open("logs.pkl", "rb"))"
"""
import os
import pandas as pd
import dill
from utils.case_directory import CaseDirectory

d = os.path.dirname(os.path.abspath(__file__))
key = pd.read_csv(f"{d}/fed_key.csv")
key["result"] = ""
logs = []

for i, row in key.iterrows():
    path = f"{d}/../data/{row.metadata_path}"
    key.loc[i, "result"], log = CaseDirectory.categorize_from_metadata_path(path)
    logs.append(log)

with open(f"{d}/logs.pkl", "wb") as f:
    dill.dump(logs, f)

key.to_csv(f"{d}/results.csv")
