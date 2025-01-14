# Mirror Detection via Multi-Directional Similarity Perception and Spectral Saliency Enhancement
Mirror Detection via Multi-Directional Similarity Perception and Spectral Saliency Enhancement

# Get started
## Datasets
The mirror segmentation dataset can be downloaded [here](https://drive.google.com/drive/folders/1Fj0fIwn-mXI3xTlENiHXjYNLMUBRTZwg?usp=sharing)

## Requirements
```
conda create -n <yourenv_name> python=3.7
conda activate <yourenv_name>
pip3 install torch torchvision torchaudio
pip3 install openmim
mim install mmcv==1.7.1
pip install -e .  # or "python setup.py develop"
pip install -r requirements/optional.txt
```
Please refer to [here](https://github.com/ZhouYanzhao/ORN/tree/pytorch-v2) for the installation of Oriented Response Networks (ORN) related environments.

The pretrained weights of Swin-s can be downloaded [here](https://pan.baidu.com/s/1p_cJWWzrKN6rke1_U1f_qA?pwd=2pf3).

## Train
```
python tools/train.py /configs/m2sd/m2sd_msd.py --load-from pretrain_checkpoint.pth
```

## Test
```
python tools/test.py /configs/m2sd/m2sd_msd.py ./checkpoint.pth --eval mIoU
```

## Visualization
```
python tools/test.py /configs/m2sd/m2sd_msd.py ./checkpoint.pth --show --show-dir save_path
```
# Results

## Evaluation on benchmark datasets

| Dataset | IoU | F | MAE | 
| :---: | :---: | :---: | :---: 
| MSD | 87.11 | 0.936 | 0.032 |  
| PMD | 69.77 | 0.846 | 0.024 | 
| RGBD-Mirror | 78.60 | 0.904 | 0.030 |

## Visualization of segmentation results

The segmentation results on test datasets can be downloaded [here](https://pan.baidu.com/s/15d3J73Se_xC-FL6EUgp9MA?pwd=hi2q)


