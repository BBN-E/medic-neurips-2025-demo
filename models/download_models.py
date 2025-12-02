#!/usr/bin/env python3

import subprocess

models = {
    # "Mistral-7B-Instruct-v0.1-GPTQ": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1",
    # "Llama3-ChatQA-1.5-8B": "https://huggingface.co/nvidia/Llama3-ChatQA-1.5-8B",
    "BioMistral-7B": "https://huggingface.co/BioMistral/BioMistral-7B",
    # "": "https://huggingface.co/google/medgemma-4b-it",

    # These two models should be downloaded automatically by our code. Uncomment if manual download is required.
    #"gpt2-large": "https://huggingface.co/openai-community/gpt2-large",
    #"all-MiniLM-L6-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",

    # BBN used this model in some of our experimentation
    #"Qwen2.5-72B-Instruct": "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct" # Optional: uncomment to include
}

try:
    subprocess.run(["git", "lfs", "install"], check=True)
    print(f"Successfully installed git lfs.")
except subprocess.CalledProcessError as e:
    print(f"Error installing git lfs: {e}")

# Loop through the dictionary and process each URL
for name, url in models.items():
    print(f"Downloading {name} from {url}")

    if name == "":
        # Prompt user to authorize access to MedGemma before cloning
        auth_url = "https://huggingface.co/google/medgemma-4b-it"
        print(f"Before attempting to download MedGemma, please authorize access by visiting {auth_url} and clicking 'Agree and access repository'")
        input("Press Enter once you have completed authorization...")

        print("When prompted for a username / password, use your HuggingFace username and, for the password, use an access token with write permissions. To generate one from your account, visit https://huggingface.co/settings/tokens")

    try:
        subprocess.run(["git", "clone", url], check=True)
        print(f"Successfully cloned {name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning {name}: {e}")
    
    print("Done processing", name)
    print("------------------------")
