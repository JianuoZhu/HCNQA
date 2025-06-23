<!--
HCNQA: Enhancing 3D VQA with Hierarchical Concentration Narrowing Supervision  
README – last update: 2025-06-23
-->

<h1 align="center">HCNQA: Enhancing 3D VQA with Hierarchical Concentration Narrowing Supervision</h1>

<p align="center">
  <a href="https://icann2025.org/accepted.html">
    <img src="https://img.shields.io/badge/ICANN-2025-ff69b4.svg" alt="ICANN 2025">
  </a>
</p>

<p align="center">
  <img src="imhs/HCNQA.png"
       alt="HCNQA architecture overview"
       width="85%">
</p>

> **TL;DR** HCNQA introduces a *three-phase Hierarchical Concentration Narrowing (HCN) supervision* strategy for 3D Visual Question Answering.  
> By explicitly supervising **coarse grounding → fine grounding → inference**, our model suppresses shortcut cues and achieves **+1.1 EM@1** and **+2.3 CIDEr** on ScanQA (*test w/ obj*) versus the previous state-of-the-art.

---


## 1 Installation
```bash
conda create -n hcnqa python=3.9 -y
conda activate hcnqa
pip install -r requirements.txt
cd model\vision\pointnet2
pip install .
```

## 2 Data Preparation
1. Please refer to the tutorial of [3D-VisTA](https://github.com/3d-vista/3D-VisTA?tab=readme-ov-file), download and extract the ScanQA dataset.
2. download the annotation of our work from [link]().

## 3 CKPT Download
1. Download the ckpt for language encoder(bert-base-uncased).
2. Download the ckpt for our model from [link]().

## 3 Training
```bash
python run.py --config project/vista/scanqa_train.yml
```

## 3 Eval
```bash
python run.py --config project/vista/scanqa_eval.yml
```

## 4 License & Acknowledgements
This repository is released under the MIT License (see LICENSE).
The whole codebase **inherits heavily from the open-sourced
<a href="https://github.com/3d-vista/3D-VisTA">3D-VisTA project</a>; 
we gratefully acknowledge their clean design and utilities.