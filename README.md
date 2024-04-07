# Article title

See related [Publications](https://github.com/janezlapajne/manuscripts)

### 🔍 Introduction


**Abstarct** \
place abstract here

**Authors:** Author1, Author2, Author3 \
**Keywords:** Keyword1, Keyword2, Keyword3 \
**Published In:** Journal or Conference Name \
**Publication Date:** Month, Year 

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

Download the data from [Zenodo](10.5281/zenodo.7936850) and unzip to folder named `imagings`.
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

This project was initially developed by Janez Lapajne. If you have any questions or encounter any other problem, feel free to post an issue on Github.
