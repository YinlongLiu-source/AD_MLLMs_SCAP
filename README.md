# AD_MLLMs_SCAP

The resource of paper **"Cross-Lingual Alzheimer's Disease Detection with Multimodal LLMs via Speech Cue-Augmented Prompting and Instruction Tuning"**.

# Environment Setup

To set up the required environment, please install the dependencies from the `requirements.txt` file.

```
pip install -r requirements.txt
```

# Dataset

To obtain the public datasets used in our study, please visit their respective official websites:

* **ADReSS:** [https://talkbank.org/dementia/ADReSS-2020/index.html](https://talkbank.org/dementia/ADReSS-2020/index.html)
* **PROCESS:** [https://processchallenge.github.io/](https://processchallenge.github.io/)

Please note that the iFLYTEK dataset is a private collection and is not publicly available at this time.

We provide `train_example.jsonl` and `test_example.jsonl` as data samples to demonstrate the required format for instruction fine-tuning and inference.

# MLLMs

Please download the pre-trained Multimodal Large Language Models (MLLMs) from the HuggingFace Hub and save them in the `MLLMs/` directory.

# Training

Our model training is implemented using the [MS-Swift](https://github.com/modelscope/ms-swift) framework.

* **`train.sh`**: This is the main script for instruction fine-tuning the MLLMs. You can customize the training process by modifying the parameters within this script.
* **`merge.sh`**: After fine-tuning, run this script to merge the LoRA adapter weights with the original model weights to create the final adapted model.

# Inference

The `inference.py` script is used to perform inference on the test data.
