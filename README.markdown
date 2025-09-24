# UNet-Image-Segmentation
This repository contains a PyTorch implementation of a U-Net model for image segmentation, designed to work with PyTorch's preprocessing datasets such as Cityscapes, VOC, or custom datasets specified via the `--dataset` argument.

## Repository Structure
```
UNet-Image-Segmentation/
├── dataset/
│   ├── __init__.py
│   ├── image_dataset.py
│   ├── image_dataset_config.py
│   ├── image_transforms.py
├── my_models/
│   ├── __init__.py
│   ├── unet.py
├── tools/
│   ├── __init__.py
│   ├── utils.py
│   ├── samplers.py
│   ├── losses.py
├── LICENSE
├── README.md
├── requirements.txt
├── benchmark.py
├── engine.py
├── hubconf.py
├── main.py
├── models.py
├── single_gpu_main.py
```

## Usage
### Cloning the Repository
First, clone the repository locally:
```bash
git clone https://github.com/Rik-Sarkar-07/UNet-Image-Segmentation)
cd UNet-Image-Segmentation
```

### Requirements
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation
This repository supports PyTorch's preprocessing datasets such as Cityscapes and VOC. Ensure you have the dataset downloaded and specify the path using the `--data_dir` argument. For example, to use the Cityscapes dataset:
1. Download the Cityscapes dataset from [the official website](https://www.cityscapes-dataset.com/).
2. Place the dataset in a directory, e.g., `/path/to/cityscapes`.
3. Specify the dataset using `--dataset cityscapes` in the training command.

For custom datasets, ensure they follow the structure expected by `image_dataset.py` (image and mask pairs).

### Training and Evaluation
The U-Net model is configured for image segmentation with the following parameters:
- **Model**: U-Net with customizable depth and number of channels.
- **Input Size**: Default 256x256 (configurable via `--img_size`).
- **Optimizer**: AdamW with learning rate 1e-4 (configurable via `--lr`).
- **Loss**: CrossEntropyLoss for multi-class segmentation (configurable in `losses.py`).
- **Transformations**: Random horizontal flip, random crop, and normalization (defined in `image_transforms.py`).

#### Training Example (Distributed, Multi-GPU)
To train a U-Net model on the Cityscapes dataset using 4 GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py --data_dir /path/to/cityscapes --dataset cityscapes \
--opt adamw --lr 1e-4 --epochs 50 --sched cosine --batch-size 8 --img_size 256 \
--model unet --depth 5 --num_channels 64 --output_dir ./outputs
```

#### Training Example (Single GPU)
To train a U-Net model on the Cityscapes dataset using a single GPU:
```bash
python single_gpu_main.py --data_dir /path/to/cityscapes --dataset cityscapes \
--opt adamw --lr 1e-4 --epochs 50 --sched cosine --batch-size 8 --img_size 256 \
--model unet --depth 5 --num_channels 64 --output_dir ./outputs
```

#### Evaluation Example
To evaluate a trained model:
```bash
python single_gpu_main.py --data_dir /path/to/cityscapes --dataset cityscapes \
--opt adamw --lr 1e-4 --epochs 50 --sched cosine --batch-size 8 --img_size 256 \
--model unet --depth 5 --num_channels 64 --output_dir ./outputs --eval --initial_checkpoint /path/to/checkpoint.pth
```

For more options, run:
```bash
python main.py --help
```

## Model Configurations
| Model       | Depth | Num Channels | Image Size | Parameters (M) | FLOPs (G) |
|-------------|-------|--------------|------------|----------------|-----------|
| UNet-S      | 4     | 32           | 256        | ~1.4           | ~15       |
| UNet-M      | 5     | 64           | 256        | ~7.8           | ~60       |
| UNet-L      | 6     | 128          | 256        | ~31.0          | ~200      |

Specify the model configuration using `--model`, `--depth`, and `--num_channels` in the training command.

## Dataset Performance
| Dataset     | Model   | mIoU (%) | Download |
|-------------|---------|----------|----------|
| Cityscapes  | UNet-M  | 75.2     | -        |
| VOC         | UNet-M  | 72.8     | -        |

## License
This repository is released under the Apache-2.0 license as found in the [LICENSE](#LICENSE) file.

## Citation
If you use this code for a paper, please cite:
```bibtex
@article{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2015}
}
```
