For basic replication of results
================================
Requirements
------------
CUDA-compatible GPU, 16GB of RAM or greater, Red Hat Enterprise Linux version 8.10 (Ootpa), Python 3.12.9. 

We typically reserve 6 CPU cores per GPU. For example CPU data, see the "Additional info" section below.

To use GPUs, make sure that the NVIDIA drivers are installed and that the available GPUs are visible to
the running Python environment using the CUDA_VISIBLE_DEVICES environment parameter. 

To download models from HuggingFace, make sure that git LFS is installed and enabled for your git install.
The models/download_models.py script is included as an example helper script.

Additional info
---------------
At BBN we tested this notebook on a minimal configuration of:
- one A6000 GPU, one Intel Xeon Gold 6128 6-core 3.40GHz processor, 16Gb RAM
- one V100 GPU, one Intel Xeon Gold 6148 20-core 2.40GHz, 16Gb RAM

For experiments with larger models
==================================
Some of the larger models require more GPUs and VRAM. See the README.md for more info on how these models are used and where to download them.

Qwen2.5-72B-Instruct Requirements
--------------------
At least 4 CUDA-compatible GPUs, 48GB RAM, AMD EPYC 7302 16-core 3.0GHz processor, Red Hat Enterprise Linux version 8.10 (Ootpa), Python 3.12.9.

Run with 6 CPU cores available per GPU.
