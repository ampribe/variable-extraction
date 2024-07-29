## Description
This project extracts variables from court documents using retrieval-augmented generation. The project is organized into the following structure:

```bash
├── extractors
│   ├── __init__.py
│   ├── bench_ruling_classifier.py
│   ├── document_summarizer.py
│   ├── extractor_config.py
│   ├── extractor_log.py
│   ├── jury_ruling_classifier.py
│   ├── trial_type_classifier.py
│   └── variable_extractor.py
├── tests
│   ├── __init__.py
│   ├── fed_key.csv
│   └── test_fed.py
├── utils
│   ├── __init__.py
│   ├── case_directory.py
│   ├── case_metadata.py
│   └── document.py
├── README.md
└── requirements.txt
```

case_directory defines CaseDirectory, which provides general methods for handling a directory of cases. Individual case information is stored in a CaseMetadata object. The variable_extractor module defines VariableExtractor, a base class that requires ExtractorConfig and CaseMetadata objects to extract a variable from a court case. 

## Installation
1. Clone the repository
`git clone https://github.com/ampribe/variable-extraction.git`
2. Create and activate a virtual environment within the project directory
`cd variable-extraction`
`python3 -m venv venv`
`.\venv\Scripts\activate`
3. Install requirements
`pip install -r requirements.txt`
4. Install necessary language models (llama3, mxbai-embed-large)
`ollama pull llama3`
`ollama pull mxbai-embed-large`

## Usage
