# TomoSeg: A Semantic Segmentation Framework for Tomography Data

## Introduction

This repository is a PyTorch implementation for semantic segmentation.
It can be used for training and testing on any type of image data, but 
is originally developed to work with tomography data.

## Models Supported
- High Resolution Network (HRNet):
  - [Paper](https://arxiv.org/abs/1902.09212)
  - [Code]()  

## Requirements
- Python 3.8.0
- PyTorch 1.11.0

## Setup
- Clone the repository: `git clone https://github.com/bashirkazimi/tomoseg.git`
- Go to directory and install required libraries:
  - `cd tomoseg`
  - `pip install -r requirements.txt`
  
## Train
- Edit `configs/hrnet_tomo.yaml` or create a new config file tailored to 
  your own dataset and desired model. For details, please check `tomoseg/config/default.py`
- Start training using the following command:

  - Using the config file (`1` gpu by default):

    ```
    python train.py --cfg configs/hrnet_tomo.yaml
    ```

  - Using 2 gpus:

    ```
    torchrun --nproc_per_node=2 train.py --cfg configs/hrnet_tomo.yaml GPUS 0,1
    ```
- The training output and model files are by default saved under `output` directory
unless it's changed in the config file.
- The tensorboard log files by default under `log` directory unless changed in 
  the config file

## Evaluate
```
python test.py --cfg configs/hrnet_tomo.yaml TEST.MODEL_FILE path/to/model/weights
```

## Predict
```
python predict.py --cfg configs/hrnet_tomo.yaml TEST.MODEL_FILE path/to/model/weights \
DATASET.UNLABELED_SET path/to/file/with/image/list.txt TEST.SV_DIR path/to/save/predictions
```

## Citation
Please cite if you find it useful:

```
@misc{tomoseg2022,
  author={Kazimi, Bashir},
  title={TomoSeg: A Semantic Segmentation Framework for Tomography Data},
  howpublished={\url{https://github.com/bashirkazimi/tomoseg}},
  year={2022}
}
```

## Questions/Suggestions?
Please feel free to send pool requests or make suggestions. You can 
reach me at `kazimibashir907 at gmail.com`

  


