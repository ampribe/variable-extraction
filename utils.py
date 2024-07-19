import os
import json
import numpy as np
import pandas as pd
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

class RAG:
    def __init__(self):
        self.client = chromadb.Client()
        
    @staticmethod
    def get_case_key(case):
        return str(abs(hash(case)))

    def get_case_collection(self, case):
        key = self.get_case_key(case)
        return self.client.get_collection(key)

    def get_loaded_docs_by_case(self, case):
        collection = self.get_case_collection(case)
        return collection.get()["documents"]

    def get_relevant_case_docs(self, case, query, num_docs, model="mxbai-embed-large"):
        collection = self.get_case_collection(case)
        prompt_embedding = ollama.embeddings(
            prompt=query,
            model=model
        )["embedding"]
        return collection.query(
            query_embeddings=[prompt_embedding],
            n_results=num_docs
        )["documents"][0]
        
    def add_docs(self, case, docs, model="mxbai-embed-large"):
        print(f"Adding documents from {case} to database...")
        key = self.get_case_key(case)
        try:
            collection = self.client.create_collection(key)
        except:
            self.client.delete_collection(key)
            collection = self.client.create_collection(key)

        for i, doc in enumerate(docs):
            if len(doc) > 0:
                embedding = ollama.embeddings(model=model, prompt=doc)["embedding"]
                collection.add(
                    ids=[str(i)],
                    embeddings=[embedding],
                    documents=[doc]
                )

    # available models: mistral, phi3, llama3
    def classify_jury_ruling(self, case, n_docs=1, model="mistral"):
        sys_prompt = """
        You are an expert legal analyst. You will be given a list of excerpts from legal documents relating to a case in the United States with a decision made by a jury. Documents are separated by ||. All documents correspond to the same case. Your task is to classify the outcome of this case into one of the following categories:
        
        plaintiff
        defendant
        undetermined
        
        If the jury decided in favor of the plaintiff, classify the outcome as plaintiff.
        If the jury decided in favor of the defendant, classify the outcome as defendant.
        If the documents provided do not identify the jury verdict or the documents are ambiguous, classify the outcome as undetermined.
        
        Respond with a JSON object in the format "{"reasoning": "...", "category": "..."}" 
        If a description identifying the jury verdict is found, reasoning should be a string in the form "The description _ found in document _ shows that the jury ruled in favor of _." 
        Otherwise, the reasoning should be a string in the form, "The documents say _, which does not identify the result of the jury trial. Therefore, the case outcome is undetermined". Do not summarize the case or documents. 
        category should only include one of the following categories as a string: plaintiff, defendant, undetermined.
        """
        docs = self.get_relevant_case_docs(case, sys_prompt, n_docs)
        context_string = "||".join(docs)
        print(f"Length of context string: {len(context_string)}")
        ctx = (len(context_string) + len(sys_prompt))//3 + 50
        response = ollama.generate(
            model=model,
            prompt=context_string,
            system=sys_prompt,
            options={"num_ctx": ctx},
            format="json"
        )["response"]
        return (response, context_string)
    

class DocDF:
    def __init__(self, df, parent_folder):
        self.df = df
        self.parent_folder = parent_folder

    @staticmethod
    def get_json(path):
        with open(path) as f:
            jsn = json.loads(f.read())
        return jsn

    def get_trial_cases(self):
        trial_mask = (self.df["trial_type"] == "bench") | (self.df["trial_type"] == "jury")
        return self.df[trial_mask].case.unique()

    def get_jury_cases(self):
        jury_mask = self.df["trial_type"] == "jury"
        return self.df[jury_mask].case.unique()

    def get_docs(self, case):
        def check_subdir(path):
            docs = []
            for entry in os.scandir(path):
                if os.path.isdir(entry):
                    docs += check_subdir(entry.path)
                else:
                    with open(entry.path, errors='ignore') as f:
                        docs.append(f.read())
            return docs
        metadata_path = self.df[self.df.case == case].iloc[0].metadata_path
        parent = os.path.dirname(metadata_path)
        path = [f.path for f in os.scandir(parent) if os.path.isdir(f)][0]
        return check_subdir(path)
    
    def get_docket_report_entries(self, case):
        case_mask = self.df.case == case
        return self.df[case_mask].document_text.fillna("").str.lower().tolist()

    def get_parties_dict(self, case):
        parties = {"plaintiff": [], "defendant": []}
        metadata_path = self.df.loc[self.df.case == case, "metadata_path"].iloc[0]
        jsn = self.get_json(metadata_path)
        if "parties" in jsn:
            for party in jsn["parties"]:
                if "type" in party and "name" in party:
                    if "plaintiff" in party["type"].lower():
                        parties["plaintiff"].append(party["name"])
                    if "defendant" in party["type"].lower():
                        parties["defendant"].append(party["name"])
        return parties
    
    def get_parties_list(self, case):
        parties_dict = self.get_parties_dict(case)
        return parties_dict["plaintiff"] + parties_dict["defendant"]

    def contains_result(self, s, case):
        parties = self.get_parties_list(case)
        has_party = lambda s: "plaintiff" in s or "defendant" in s or any(map(lambda party: party.lower() in s.lower(), parties))
        has_result_keywords = lambda s: any([word in s for word in ["verdict", "judgement", "opinion", "decision", "decree", "order", "ruling", "disposition", "finding", "trial"]])
        return has_result_keywords(s) and has_party(s)
        
    @staticmethod
    def split_docs(docs, separators=["\n\n", "\n", ". ", "!", "?", ".", ";",":", ",", " ", ""], chunk_size=1000, overlap=0):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        components = []
        for doc in docs:
            components += text_splitter.split_text(doc)
        return components

    def get_relevant_chunks_from_text(self, docs, case):
        chunks = self.split_docs(docs)
        return [chunk for chunk in chunks if self.contains_result(chunk.lower(), case)]

    def get_relevant_metadata_chunks(self, case):
        docs = self.get_docket_report_entries(case)
        return self.get_relevant_chunks_from_text(docs, case)

    def get_relevant_document_chunks(self, case):
        docs = self.get_docs(case)
        return self.get_relevant_chunks_from_text(docs, case)
        
    def classify_trial_ruling_with_metadata(self, case, model):
        relevant_chunks = self.get_relevant_metadata_chunks(case)
        if len(relevant_chunks) > 0:
            print("Checking relevant docket_report entries...")
            rag = RAG()
            rag.add_docs(case, relevant_chunks)
            resp, context_string = rag.classify_jury_ruling(case, 6)
            return (json.loads(resp), context_string)
        print("No relevant docket_report entries found")
        return ({"reasoning": "No relevant metadata", "category": "undetermined"}, "")

    def classify_trial_ruling_with_documents(self, case, model):
        relevant_chunks = self.get_relevant_document_chunks(docs)
        if len(relevant_chunks) > 0:
            print("Checking relevant documents...")
            rag = RAG()
            rag.add_docs(case, relevant_chunks)
            resp, context_string = rag.classify_jury_ruling(case, 10)
            return (json.loads(resp), context_string)
        return ({"reasoning": "No relevant documents", "category": "undetermined"}, "")

    @staticmethod
    def load_log():
        if os.path.exists("logging.json"):
            with open("logging.json", "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def write_log(log):
        with open("logging.json", "w") as f:
            json.dump(log, f)
        
    def classify_trial_ruling(self, case, model="mistral"):
        logging = self.load_log()
        if case in logging:
            print("Case already classified...")
            if "document_response" in logging[case]:
                return logging[case]["document_response"]
            return logging[case]["metadata_response"]
        logging[case] = {}
        
        # 1. Check whether ruling can be identified from metadata
        print("Checking metadata...")
        resp, metadata_context = self.classify_trial_ruling_with_metadata(case, model)
        logging[case]["metadata_response"] = resp
        logging[case]["metadata_context"] = metadata_context
        if "category" in resp and resp["category"] in ["plaintiff", "defendant"]:
            self.write_log(logging)
            return resp
        print("Metadata classification unsuccessful, checking documents...")
        # 2. If ruling cannot be identified from metadata, use documents to identify ruling
        resp, docs_context = self.classify_trial_ruling_with_documents(case, model)
        logging[case]["document_response"] = resp
        logging[case]["document_context"] = docs_context
        self.write_log(logging)
        return resp