# Dataset

Place the two CSV files here:

```
data/
├── train.csv   (7,352 rows × 563 columns, 21 people)
└── test.csv    (2,947 rows × 563 columns, 9 people)
```

**Where to get them**
1. From this submission: `04_Dataset/UCI_HAR_dataset_train_test.zip` — unzip it into this folder.
2. Or from Kaggle: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones (free account, click *Download*, unzip `train.csv` and `test.csv` here).
3. Original source: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/240 (DOI 10.24432/C54S4K). The UCI archive stores the same data as text files; the Kaggle CSVs are the format our code expects.

**Columns:** 561 numerical features (time- and frequency-domain statistics of 50 Hz accelerometer and gyroscope signals, already scaled to [−1, 1]), `subject` (1–30, used only for the person-wise split) and `Activity` (one of 6 labels).

Check the files with `python -m src.data data`.

The CSVs are not committed to Git and are not uploaded to Vercel (see `.gitignore` and `.vercelignore`); the website only needs the exported files in `model/`.
