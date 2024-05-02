<h2 align="center" > (PyTorch) License Plate Detection and Recognition in Unconstrained Scenarios </h2>

<p align="center">
  <img src="https://github.com/huanidz/scaled-alpr-unconstrained/assets/78603547/9b9f0833-4263-4150-9ff7-e0dc8c69d011" alt="PyTorch Logo" width="200"/>
</p>

<h3 align="center" > PyTorch implementation + scaled of WPOD with simplified code for training + inferencing! </h3>

---

<p align="center">
  <img src="https://github.com/huanidz/scaled-alpr-unconstrained/assets/78603547/9fe7885f-8122-4dfb-b939-bfb41e7e593c"/>
</p>

<h2> 1. Environment installation (venv or conda) </h2>

<h5> Note: PyTorch 1.9.x is minimum required version </h5>
<h5> Tested with python 3.8.x </h5>

```bash
python -m pip install -r requirements.txt
```

<h2> 2. Dataset preparation </h2>
<h5> Please arrange your dataset folder like this </h5>

```
Your_Folder_Path/
├── train
│   ├── images
│   │   ├── train_image_01.png
│   │   ├── train_image_02.png
│   │   ...
│   └── labels
│       ├── train_label_01.txt
│       ├── train_label_02.txt
│       ...
└── eval
    ├── images
    │   ├── eval_image_01.png
    │   ├── eval_image_02.png
    │   ├── ...
    └── labels
        ├── eval_label_01.txt
        ├── eval_label_02.txt
        ├── ...
```

<h5> For each pair of data (image + label), the image is the original image, the label file contains coordinate of the plate with respect to the original image W and H (order is same as original repo) </h5>
<h5> Note: Currently only support one plate per image. The order of 1-->4 is (x1 - y1: top left, x2 - y2: top right, x3 - y3: bottom right, x4 - y4: bottom left) </h5>

```bash
# x1, x2, x3, x4, y1, y2, y3, y4
0.497917, 0.677083, 0.670833, 0.489583, 0.734737, 0.747368, 0.844211, 0.831579
```

<h2> 3. Training </h2>

```bash
# Simple 'base' scale training (384x384 input size, ~1.7M parameters, SGD optimizer, 200 epochs, lr = 0.001, batch_size = 16)
python train.py --data path_to_your_dataset_folder/

# Custom model scale example
python train.py --data path_to_your_dataset_folder/ --scale small --size 256 --bs 32 --lr 0.0003
```
<h5> Model checkpoints will be saved into checkpoints/ in .pth format </h5>

<h2> 4. Inference </h2>

```bash
# Example of inferencing with scale 'base', input_size = 384x384, threshold = 0.5.
python inference.py --model_path your_model.pth --size 384 --scale base --threshold 0.5
```

<h2> 5. ONNX deployment </h2>

```bash
python export_onnx.py --model_path your_model.pth --size 384 --scale small
```

---

### TODOs:
- [x] Training pipeline
- [x] Inference pipeline
- [ ] Try and update backbone/head for more accuracy
- [ ] Add pretrained weights
- [ ] Multiple GPUs training

---


```
@INPROCEEDINGS{silva2018a,
  author={S. M. Silva and C. R. Jung}, 
  booktitle={2018 European Conference on Computer Vision (ECCV)}, 
  title={License Plate Detection and Recognition in Unconstrained Scenarios}, 
  year={2018}, 
  pages={580-596}, 
  doi={10.1007/978-3-030-01258-8_36}, 
  month={Sep},}
```
