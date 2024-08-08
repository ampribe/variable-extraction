## Description
This project extracts variables from court documents using retrieval-augmented generation. The project is organized into the following structure:

```bash
├── extractors
│   ├── __init__.py
│   ├── extractor_config.py
│   ├── extractor_log.py
│   └── variable_extractor.py
├── utils
│   ├── __init__.py
│   ├── case_directory.py
│   ├── case_metadata.py
│   └── document.py
├── notebooks
│   ├── compare_trial_classification.ipynb
│   └── visualize_tests.ipynb
├── README.md
└── requirements.txt
```

The `case_directory` module defines `CaseDirectory`, which provides general methods for handling a directory of cases. Individual case information is stored in a `CaseMetadata` object. The `variable_extractor` module defines `VariableExtractor`, which requires `ExtractorConfig` and `CaseMetadata` objects to extract a variable from a court case. Default `ExtractorConfig` instances for extracting common variables can be found in `ExtractorConfig.get_config`. 

## Installation
### Prerequisites
- Python 3
- pip3
- venv `python3 -m pip install --user virtualenv`
- [Ollama](https://github.com/ollama/ollama)

### Instructions
1. Clone the repository
```
git clone https://github.com/ampribe/variable-extraction.git
```
2. Create and activate a virtual environment within the project directory
```
cd variable-extraction
python3 -m venv venv
source venv\bin\activate (MacOS/Linux)
venv\Scripts\activate (Windows)
```
3. Install requirements
```
pip3 install -r requirements.txt
```
4. Install ollama from [here](https://ollama.com/download)
5. Install necessary language models (llama3.1, mxbai-embed-large are used by default)
```
ollama pull llama3.1
ollama pull mxbai-embed-large
```

## Usage
A common error is that Jupyter may use the system Python kernel instead of Python from the virtual environment. As explained [here](https://stackoverflow.com/questions/37891550/jupyter-notebook-running-kernel-in-different-env), this can be resolved by running `python -m ipykernel install --user --name venv --display-name "venv Python"` and then selecting "venv Python" as the kernel in Jupyter. 

## Testing
To test on federal cases, create a directory `data` in the project directory and add 100_random_fed to `data`. Then, run `python3 -m tests.test_fed` from the project directory. This will create "results.csv" and "logs.pkl" files in the tests directory which can be loaded using the instructions in  `test_fed.py`.