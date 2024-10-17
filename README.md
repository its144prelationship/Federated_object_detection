# Federated-Benchmark: A Benchmark of Real-world Images Dataset for Federated Learning

## Overview

Present a real-world image dataset, reflecting the characteristic real-world federated learning scenarios, and provide an extensive benchmark on model performance, efficiency, and communication in a federated learning setting.

## Resources

- Dataset: [dataset.fedai.org](https://dataset.fedai.org)
- Paper: ["Real-World Image Datasets for Federated Learning"](https://arxiv.org/abs/1910.11089)

### Street_Dataset

- Overview: Image Dataset
- Details: 7 different classes, 956 images with pixels of 704 by 576, 5 or 20 devices
- Task: Object detection for federated learning
- [Dataset_description.md](https://github.com/FederatedAI/FATE/blob/master/research/federated_object_detection_benchmark/README.md)

## Getting Started

Implemented two mainstream object detection algorithms (YOLOv3 and Faster R-CNN). Code for YOLOv3 is borrowed from [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3.git) and Faster R-CNN from [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch.git).

### Install dependencies

- Install other dependencies, `pip install -r requirements.txt`
- Install `pip install Cython` before running
  ```bash
  cd model/utils/nms/
  python build.py build_ext --inplace
  cd -
  ```
- Using visual env by running
  ```bash
  python3 -m venv $venvname$
  source $venvname$/bin/activate
  ```

### Prepare data

1. Download the dataset, refer to [dataset.fedai](https://dataset.fedai.org/)
2. It should have the basic structure for faster r-cnn
   ```bash
   Federated-Benchmark/data/street_5/$DEVICE_ID$/ImageSets
   Federated-Benchmark/data/street_5/$DEVICE_ID$/JPEGImages
   Federated-Benchmark/data/street_5/$DEVICE_ID$/Annotations
   ```
3. Generate config file for federated learning
   ```bash
   cd data
   python3 generate_task_json.py
   ```
4. Install darknet53.conv.74
   ```bash
   cd weights
   bash download_weights.sh
   ```

### Train

1. Start server
   ```bash
   sh ./run_server.sh street_5 yolo 1234
   ```
2. Start clients
   ```bash
   sh ./run.sh street_5 5 yolo 1234
   ```
3. Stop training
   ```bash
   sh ./stop.sh street_5 yolo
   ```

### Citation

- If you use this code or dataset for your research, please kindly cite our paper:

```bash
@article{luo2019real,
  title={Real-World Image Datasets for Federated Learning},
  author={Luo, Jiahuan and Wu, Xueyang and Luo, Yun and Huang, Anbu and Huang, Yunfeng and Liu, Yang and Yang, Qiang},
  journal={arXiv preprint arXiv:1910.11089},
  year={2019}
}
```
