# Welcome to AISTrainer!

**Aistrainer** is a library built on top of **Hugging Face Transformers**, designed to simplify the process of fine-tuning large language models (LLMs) for developers. It focuses on making LLM fine-tuning feasible even with limited computational resources, such as Nvidia GeForce RTX 3090 GPUs. The library supports training on both GPUs and CPUs, and it includes a feature for offloading model weights to the CPU when using a single GPU. With one **Nvidia GeForce RTX 3090 GPU** and **256 GB of RAM**, Aistrainer can handle fine-tuning models with up to approximately **70 billion** parameters.

## Environment

The Aistrainer library is compatible with the Ubuntu 22.04 operating system. To set up the required environment for this library, system tools must be installed using the command: 
```console
sudo apt install -y python3-pip cmake mpich conda
```
To create a Python virtual environment with a GPU, use the command:
```console
conda env create -f environment.yml
``` 
In the absence of a GPU, the environment can be set up with the command: 
```console
conda env create -f environment-cpu.yml
``` 
These steps ensure that all necessary dependencies are correctly configured, allowing the Aistrainer library to function optimally.

## Updating operating system drivers
The following commands allow you to update operating system drivers:
```console
sudo rm -r /var/lib/dkms/nvidia
sudo dpkg -P --force-all $(dpkg -l | grep "nvidia-" | grep -v lib | awk '{print $2}')
sudo ubuntu-drivers install
```

## Use with JupiterLab
If you use JupiterLab then you need to add a new kernel with a conda environment:
```console
conda activate aist
conda install ipykernel
ipython kernel install --user --name=aist
```
