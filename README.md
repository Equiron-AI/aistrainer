# Welcome to AISTrainer!
**Aistrainer** is a library built on top of **Hugging Face Transformers**, designed to simplify the process of fine-tuning large language models (LLMs) for developers. It focuses on making LLM fine-tuning feasible even with limited computational resources, such as Nvidia GeForce RTX 3090 GPUs. The library supports training on both GPUs and CPUs, and it includes a feature for offloading model weights to the CPU when using a single GPU. With one **Nvidia GeForce RTX 3090 GPU** and **256 GB of RAM**, Aistrainer can handle fine-tuning models with up to approximately **70 billion** parameters.

## Environment
The Aistrainer library is compatible with the Ubuntu 22.04 operating system. To set up the required environment for this library, system tools must be installed using the command: 
```console
sudo apt install -y python3-pip ccache make cmake g++ mpich
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

## Installation
```console
pip install aistrainer
``` 

## Updating operating system drivers
The following commands allow you to update operating system drivers:
```console
sudo rm -r /var/lib/dkms/nvidia
sudo dpkg -P --force-all $(dpkg -l | grep "nvidia-" | grep -v lib | awk '{print $2}')
sudo ubuntu-drivers install
```

## Use with JupyterLab
If you use JupyterLab then you need to add a new kernel with a conda environment:
```console
conda activate aist
conda install ipykernel
ipython kernel install --user --name=aist
```

## Using swap
When fine-tuning models with a large number of parameters, it might be necessary to increase the operating system's swap space. This can be done using the following steps:

```console
sudo swapoff -a
sudo fallocate -l 50G
sudo chmod 600
sudo mkswap /swapfile
sudo swapon /swapfile
```

These commands will increase the swap space, providing additional virtual memory that can help manage the large memory requirements during model fine-tuning.

Swap should be used only in case of extreme necessity, as it can significantly slow down the training process. To ensure that the system uses swap space minimally, you should add the following line to the **/etc/sysctl.conf file**: **vm.swappiness=1**. This setting minimizes the swappiness, making the system less likely to swap processes out of physical memory and thus relying more on RAM, which is much faster than swap space.

## Convensions
- If a GPU is available, the Aistrainer library automatically leverages DeepSpeed to offload model weights to RAM. This optimization allows for efficient management of memory resources, enabling the fine-tuning of larger models even with limited GPU memory.
- The Aistrainer library supports only a specific dataset format, which must include the following columns: "instruct", "input", and "output". These columns are essential for the proper functioning of the library, as they structure the data in a way that the model can interpret and learn from effectively.
- If the eval=True parameter is passed to the prepare_dataset method, the Aistrainer library will automatically use 10% of the data in the dataset as validation data, creating an evaluation dataset. This feature allows for easy splitting of the dataset, ensuring that a portion of the data is reserved for evaluating the model's performance during training, thereby facilitating better model assessment and tuning.
- The Aistrainer library fundamentally avoids using quantization during the fine-tuning process to prevent any potential loss of quality. This approach ensures that the experiments remain straightforward and maintain the highest possible model accuracy.
- For combining LoRA adapters, the Aistrainer library supports only the "cat" method. In this method, the LoRA matrices are concatenated, providing a straightforward and effective approach for merging adapters.

## Supported Models
The following LLM models are supported:
- Phi-3-medium-128k-instruct
- c4ai-command-r-v01

## Training example
```python
import logging
from aistrainer.aistrainer import Aist

logging.basicConfig(level=logging.INFO)

aist = Aist("CohereForAI/c4ai-command-r-v01")

aist.prepare_dataset("equiron-ai/safety",
                     eval=False,  # use the entire dataset only for training
                     max_len_percentile=100)  # percentile cutting off the longest lines

aist.train("safety_adapter",
           rank=16,
           lora_alpha=32,
           batch_size=4,  # suitable for most cases, but should be reduced if there is not enough GPU memory
           gradient_steps=2)  # suitable for most cases, but should be reduced if there is not enough GPU memory
```

## Combining/merging LoRA adapters
```python
import logging
from aistrainer.aistrainer import Aist

logging.basicConfig(level=logging.INFO)

aist = Aist("CohereForAI/c4ai-command-r-v01")
aist.merge("model_with_safety", "safety_adapter")
```

## Known issues
Model fine-tuning and combining adapters cannot be performed in the same bash script or Jupyter session. It is essential to separate the processes of fine-tuning and adapter merging. When using JupyterLab, you must restart the kernel after completing each of these processes to ensure proper execution and avoid conflicts.

## Convert to GGUF
```console
python3 llama.cpp/convert-hf-to-gguf.py /path/to/model --outfile model.gguf --outtype f16
llama.cpp/build/bin/quantize model.gguf model_q5_k_m.gguf q5_k_m
```

## Run with Llama.CPP Server on GPU
```console
llama.cpp/build/bin/server -m model_q5_k_m.gguf -ngl 99 -fa -c 4096 --host 0.0.0.0 --port 8000
```

## Install CUDA toolkit for Llama.cpp compilation
Please note that the toolkit version must match the driver version. The driver version can be found using the nvidia-smi command.
Аor example, to install toolkit for CUDA 12.2 you need to run the following commands:
```console
CUDA_TOOLKIT_VERSION=12-2
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-${CUDA_TOOLKIT_VERSION}
echo -e '
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
' >> ~/.bashrc
```

