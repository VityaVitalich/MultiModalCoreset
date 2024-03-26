# MultiModalCoreset

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
- notebooks/ directory contains notebooks used for debugging and small experiments
- tests/ directory contains pytest testing for the multimae part

### Dataset Quantization

Please see README.md at Dataset_Quantization/ directory

### Bot

bot/ directory represents code for TG bot. 

- bot.py is the main script to run
- messages.py includes system messages that are used in replies
- model_utils.py includes functions for loading and infering MultiMAE
- rate_handler.py includes code for rating MultiMAE performance
- system_handlers.py includes code for handling system messages
- aiogram_tests includes library for testing
- tests.py includes tests for bot
- htmlcov shows tests coverage
- figs directory contains required images for correct bot functioning

### FastAPI

Please see fastapi/ dir for details

### Docker

The docker image for the whole project located [here](https://hub.docker.com/r/vityavitalich/dq_image). It also contains checkpoints for main models, obligatory for running any inference. The image was created from Dockerfile with only execption that checkpoints are located at fastapi/ckpt/

## Dataset Statistics

#### CLEVR
1. 70K RGB Images
2. Multiple Modalities Avaliable
3. Validation 15K Images
4. 512x512 images
5. Mean pixel intensity (121.21886965, 120.20393113, 118.47079931)
6. Std of pixel intensity (22.98711599, 22.20778617, 22.16254607)

![Pixel intensity distribution](https://github.com/VityaVitalich/MultiModalCoreset/blob/main/bot/figs/pixel_intensity.png)


#### CIFAR 10

1. 60K RGB Images 
2. 32x32
3. 10 classes
4. 6K per class
5. 50K Train, 10K Test

#### Imagenet 1K
1. Train 1,281K Images
2. Validation 50K Images
3. 1000 Classes
