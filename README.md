# Challenges of Multi-Modal Coreset Selection for Depth Prediction


This repository contains code for paper [Challenges of Multi-Modal Coreset Selection for Depth Prediction](https://arxiv.org/abs/2502.15834) from ICLR 2025 ICBINB Workshop.

## Replicating Experiments

The repository is splitted into part with training/fine-tuning and selecting coreset. 

Multimae dir is capable of training/fine-tuning and obtaining embeddings. Please see Usage Section.

To obtain every model described in a paper, only modifying the config file is enough.


Dataset Quantization directory is capable of coreset selection. Please see Usage Part about it


## Usage

### MultiMAE

- Located at multimae/
- run_finetuning_depth.py performs fine-tuning of MultiMAE to depth prediction task
- It uses configs from configs/depth.py
- Configs are used for changing paths, input domains, size of patches, training parameters and logging
- Training performed with Trainer class. Base trainer is located at base_trainer.py, it describes main functionality like train step, validation, etc
- However it lacks some methods special for every out domain, therefore specific DepthTrainer is inherited from BaseTrainer and located at rgb_d_trainer.py
- dataset.py contains code for dataset collection and proccessing
- input_adapters.py contains adapters for input domains
- output_adapters.py and output_adapter_utils.py contain code for output adapters
- mae_models.py, multimae.py and multimae_utils.py contain code for main multiMAE model


To obtain embeddings one needs to run `obtain_embedding.py` script. The script is controlled via the same config as training.

To use PCA or UMAP one needs to run `reduce_dimension.py` scriptw with arguments stated there. 
### Dataset Quantization

The original README is in the directory

- all the source code is in coreset/ dir
- to run the sampling one can see example in `run_sample_multi.sh` script


 
