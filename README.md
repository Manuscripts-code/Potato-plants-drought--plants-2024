# Enhancing Water-Deficient Potato Plant Identification: Assessing Realistic Performance of Attention-Based Deep Neural Networks and Hyperspectral Imaging for Agricultural Applications

See related [Publications](https://github.com/janezlapajne/manuscripts)

### 🔍 Introduction

**Abstract**

Hyperspectral imaging has emerged as a pivotal technology in agricultural research, offering a powerful means to non-invasively monitor stress factors, such as drought, in crops like potato plants. In this context, the integration of attention-based deep learning models presents a promising avenue for enhancing the efficiency of stress detection, by enabling the identification of meaningful spectral channels. The study assesses performance of deep learning models on two potato plant cultivars exposed to water-deficient condition. It explores how various sampling strategies and biases impact the classification metrics by using a dual-sensor hyperspectral imaging system (VNIR and SWIR). Moreover, it directed its focus towards pinpointing crucial wavelengths within the concatenated images indicative of water-deficient condition. The proposed deep learning model yields encour-aging results. In the context of binary classification, it achieved an area under the receiver operating characteristic curve (AUC-ROC) of 0.74 (95% CI: 0.70, 0.78) and 0.64 (95% CI: 0.56, 0.69) for KIS Krka and KIS Savinja varieties, respectively. Moreover, the corresponding F1 scores were 0.67 (95% CI: 0.64, 0.71) and 0.63 (95% CI: 0.56, 0.68). The evaluation of performance on datasets with deliberately introduced biases consistently demonstrated superior results in comparison to their non-biased equivalents. Notably, the ROC-AUC values exhibited significant improvements, registering a maximum increase of 10.8% for KIS Krka and 18.9% for KIS Savinja. The wavelengths of greatest significance were observed in the ranges of 475 – 580 nm, 660 – 730 nm, 940 – 970 nm 1420 – 1510 nm, 1875 – 2040 nm, and 2350 – 2480 nm. These findings suggest that discerning between the two treatments is attainable, despite the absence of prominently manifested symptoms of drought stress in either cultivar through visual observation. The research outcomes carry significant implications for both precision agriculture and potato breeding. In precision agriculture, precise water monitoring enhances resource allocation, irrigation, yield, and loss prevention. Hyperspectral imaging holds potential to expedite drought-tolerant cultivar selection, thereby streamlining breeding for resilient potatoes adaptable to shifting climates.

**Authors:** Janez Lapajne*, Ana Vojnović, Andrej Vončina and Uroš Žibrat \
**Keywords:** Hyperspectral imaging; deep learning; potato plant; water-deficiency; drought stress \
**Published In:** [Plants](https://www.mdpi.com/2223-7747/13/14/1918) \
**Publication Date:** 07/2024

### ⚙️ Environment setup

**Requirements (desired)**

* 🎮 Sufficiently powerful GPU, min. 4GB VRAM
* 💾 Min. 64 GB RAM
* 📦️ Min. 100 GB available storage memory

**Local setup**

Setup is written for Windows machine. However, the same setup is required for Linux machine.

1) Create and activate a virtual environment:

```bash
conda create -n env-eval python=3.9
conda activate env-eval
```

2) Install packages into the virtual environment:

```bash
pip install -r requirements.txt
```

3) Install Pytorch CUDA support if not automatically installed.

### 🖼️ Dataset

Download the data from [Zenodo](https://zenodo.org/records/7936850) and unzip to folder named `imagings`.
The folder structure should look like:

```
📂 imagings
├── 📁 imaging-1
│   ├── 📄 0_1_0__KK-K-04_KS-K-05_KK-S-03__imaging-1__1-22_20000_us_2x_HSNR02_ 2022-05-11T104633_corr_rad_f32.hdr
│   ├── 📄 0_1_0__KK-K-04_KS-K-05_KK-S-03__imaging-1__1-22_20000_us_2x_HSNR02_2022-05-11T104633_corr_rad_f32.img
│   └── 📄 ...
├── 📁 imaging-2
│   └── 📄 ...
├── 📁 imaging-3
│   └── 📄 ...
├── 📁 imaging-4
│   └── 📄 ...
└── 📁 imaging-5
    └── 📄 ...
```

Then, create `.env` file in repository root (next to `.env.example`) and specify the **absolute** path to extracted data location.
For example, if the data is located in `C:\\Users\\janezla\\Documents\\imagings`, write the following in the `.env` file (without spaces and unusual characters):

```sh
DATA_DIR=C:\\Users\\janezla\\Documents\\imagings
```

### 📚 How to use

**Train and evaluate**

Run the following command to train the model on training data and evaluate on testing data.

```bash
python main.py -c configs/krka/stratify/krka_stratify_54321.json -m train_test
```

Use different json config file accordingly.

**Observe experiments**

The experiments are automatically created by using mlflow tool. To start mlflow server run:

```bash
mlflow server -h 0.0.0.0 -p 8000 --backend-store-uri experiments/
```

The experiments could than be reached at <http://localhost:8000/>

**Generate results**

Use scripts and notebooks from `notebooks` directory to generate results, plots and classification metrics.
For example, run `produce_results.py` script to generate the metrics and some results.

### 📬 Contact

This project was initially developed by [Janez Lapajne](https://github.com/janezlapajne). If you have any questions or encounter any other problem, feel free to post an issue on Github.
