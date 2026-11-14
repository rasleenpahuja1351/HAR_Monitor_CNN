# Results (test set: 2,947 windows from 9 unseen subjects)

| Model | Test accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Validation accuracy | Size |
|---|---|---|---|---|---|---|
| Dense NN | 93.5% | 0.942 | 0.934 | 0.935 | 97.1% | 455,430 params |
| Random Forest | 92.9% | 0.930 | 0.926 | 0.927 | 93.1% | 300 trees |
| 1D CNN | 89.1% | 0.906 | 0.891 | 0.890 | 92.1% | 53,222 params |

## Per-class F1

| Activity | Dense NN | Random Forest | 1D CNN |
|---|---|---|---|
| LAYING | 0.988 | 1.000 | 0.978 |
| SITTING | 0.867 | 0.899 | 0.839 |
| STANDING | 0.887 | 0.909 | 0.816 |
| WALKING | 0.950 | 0.936 | 0.911 |
| WALKING_DOWNSTAIRS | 0.959 | 0.909 | 0.881 |
| WALKING_UPSTAIRS | 0.960 | 0.909 | 0.914 |
