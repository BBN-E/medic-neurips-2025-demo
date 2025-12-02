This is the codebase for the NeurIPS 2025 Demo: Statistically Significant Results 
On Biases and Errors of LLMs Do Not Guarantee Generalizable Results.
https://arxiv.org/abs/2511.02246v1

## Dataset

The dataset associated with this codebase, including commands we ran to generate it,
may be downloaded from here: https://zenodo.org/records/17663093

## Getting Started

### Requirements

#### Hardware

To run MEDIC, you will need one or more GPUs, generally with at least 16GB RAM per GPU. BBN has
successfully replicated MEDIC results using the following GPU architectures: A6000, L40, P100, V100 (16G and 32G).

For a summary of hardware and software requirements, see system_requirements.md.

#### Software

MEDIC was built and tested using Python 3.12. The full list of Python packages it uses is
documented in the following section "Installing Python Dependencies." Additionally, you will
need to install the LLMs and related models used by MEDIC (see "Installing Models"), plus a local
version of https://github.com/oobabooga/text-generation-webui (details below).

### Installing Python Dependencies

To build our Python environment using the following recipe (and to replicate our reported performance and
results), we require Python 3.12 and PyTorch 2.6.0 with GPU and CUDA support. If you attempt to build the 
following environments on a machine that does not have a GPU or CUDA, it will likely fail to resolve all
the dependencies.

For additional information on OS and hardware requirements, see system_requirements.md.

#### Using Pip

To create a virtual env in which to load the dependencies, run:

    # create a virtualenv in the current directory called 'venv'. For a different name, change the second venv.
    python3.12 -m venv venv

    # this activates the virtualenv created in the last line. If you changed the 'venv' there, update here too.
    source venv/bin/activate

    # the latest version of pip is required in order to find some of the newer dependencies
    pip install --upgrade pip   # make sure using pip 25.0.1

If your environment is a direct match to ours, then you may be able to load the full, exact Python requirements.txt
(created using pip freeze > requirements.txt) with the following command:

    pip install -r requirements.txt

If this throws an error (likely due to a difference such as operating system or other platform-specific difference),
a much shorter list of direct dependencies used by our project are documented in the requirements.in. You can
instantiate a python environment with this file using the following command:

    pip install -r requirements.in

This should load many of the same dependencies as listed in requirements.txt; note any differences by running

    pip freeze > requirements-local.txt
    diff requirements.txt requirements-local.txt

Once you have installed the python packages using either the requirements.txt or requirements.in files,
install the en_core_web_sm model for the spacy package (described below) by running

    python -m spacy download en_core_web_sm

#### Using Conda (Optional)

As an alternative to pip, some python users prefer to use conda. If you have already installed the
python dependencies using pip in the previous step, feel free to skip this section.

To install the python dependencies using the open-source conda tool, select the appropriate version 
of conda dependency yml file from runtime_environments/*.yml and then run the following command:

    conda env create -f runtime_environments/conda/medic-031225.yml

Once the conda environment has been created, activate the environment, then 
install the en_core_web_sm model for the spacy package (described below) by running

    python -m spacy download en_core_web_sm


#### Loading the bbn_medic python code into the python environment

One package not included in the above pip or conda environments is the bbn_medic python codebase itself.
This is because at BBN, multiple users share a project's versioned conda environments. Keeping the bbn_medic
project codebase out of those shared environments (and adding it at invocation) allows reuse of the environment
with differing versions of MEDIC to support developing and testing different features. 

The remainder of our documentation will preface calls to python with a PYTHONPATH=<path/to/medic/src/python/directory>
to ensure that the PYTHONPATH includes the parent directory of the bbn_medic package.

For someone who doesn't resonate with the above concerns and prefers to install MEDIC into the python
environment directly, it is possible using the following command:

    pip install -e src/python

If you do this, the PYTHONPATH=<path> prefaces to invocations of python should no longer be necessary. 

### Installing Models

MEDIC requires the following models to run its default configurations. These are not included in the delivery; however,
a helper script to download them from HuggingFace is found in ./models/download_models.py. Please note that, in order
to download the Mistral-7B model from HuggingFace, you will need to set up a token following the instructions at
https://huggingface.co/docs/hub/security-tokens#user-access-tokens .


| Model                         | URL | What MEDIC uses it for                                                             | Requirements                         |
|------------------------------ | --- |------------------------------------------------------------------------------------|--------------------------------------|
| Mistral-7B-Instruct-v0.1-GPTQ | https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 | Desire / prompt generation, question answering, hallucination / omission detection | Recommend using at least 16GB RAM per GPU |
| Llama3-ChatQA-1.5-8B | https://huggingface.co/nvidia/Llama3-ChatQA-1.5-8B | Desire / prompt generation, question answering, hallucination / omission detection | Recommend using at least 16GB RAM per GPU |
| BioMistral-7B | https://huggingface.co/BioMistral/BioMistral-7B |  Desire / prompt generation, question answering, hallucination / omission detection | Recommend using at least 16GB RAM per GPU |
| MedGemma-4B |  https://huggingface.co/google/medgemma-4b-it |  Desire / prompt generation, question answering, hallucination / omission detection | Recommend using at least 16GB RAM per GPU |
| en_core_web_sm | https://spacy.io/models/en | Answer manipulation; required by spacy python package | Install with python (see above) |
| gpt2-large | https://huggingface.co/openai-community/gpt2-large | Perplexity metric (should be downloaded automatically by our code)           | Recommend using 16GB RAM per GPU         |
| all-MiniLM-L6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Faithfulness metric  (should be downloaded automatically by our code)    | Recommend using 16GB RAM per GPU         |


Additional models you may wish to download and experiment with:

| Model                         | URL | Purpose | Requirements |
|------------------------------ | --- | ------- | ------- |
| Qwen2.5-72B-Instruct | https://huggingface.co/Qwen/Qwen2.5-72B-Instruct | Desire / prompt generation, question answering, hallucination / omission detection | at least 4 GPUs and 48GB memory |

### Installing text-generation-webui (required for agentic hallucination and omission detectors)

MEDIC's baseline hallucination and omission detectors use huggingface models via langchain, but the
agentic workflow uses autogen to talk to models via an API. For the API interface to LLMs, we use 
text-generation-webui. Therefore, to run the agentic hallucination and omission detectors, you will 
need to perform the following:

1) Download and install from https://github.com/oobabooga/text-generation-webui. We specifically recommend
using the version from git commit 769eee1 (as this is what we tested with).

2) Modify line 36 of src/python/bbn_medic/detection/agentic/utils/llm_service.py:

       cwd="/nfs/nimble/projects/hiatus/hqiu/frozen/text-generation-webui/769eee1",

replacing the path above with the path to your local instance of text-generation-webui.

### Tests

After you've installed MEDIC's Python dependencies, you can run the project's Python unit test suite from the 
src/python/bbn_medic directory by running

    pytest src/python/bbn_medic/tests/unit

If you have a GPU and LLMs installed per the above instructions, it is possible to run the additional integration
tests found in src/python/bbn_medic/tests. However, currently these tests (in src/python/bbn_medic/*) are 
hard-coded to point to models at path on BBN's HPC infrastructure. They would need to be modified on a case-by-case 
basis.

## License

This project is licensed under the Apache 2 License (https://www.apache.org/licenses/LICENSE-2.0).
See the LICENSE.txt file for details.

## Acknowledgments

This research was, in part, funded by the Advanced Research Projects Agency for Health (ARPA-H). The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Government.
