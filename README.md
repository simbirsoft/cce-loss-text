# Experiment on applying complement cross-entropy loss to text classification problem

## Steps to reproduce
1. Install dependencies:

```shell
pip install -r requirements.txt
```

2. Donwload and extract data from [Kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/overview)

3. Preprocess the data:

```shell
python preprocess_data.py <path-to-train.tsv> <path-to-dataset-folder>
```

4. Run experiment with standard cross-entropy loss:

```shell
python run_experiments.py <path-to-dataset-folder>
```

Add ```--gpu``` to train on GPU

5. Run experiment with complement cross-entropy loss:

```shell
python run_experiments.py <path-to-dataset-folder> --use_cce
```

6. Accuracy scores will appearn on screen. More details will be available in Tensorboard:

```shell
tensorboard --logdir=./runs
```