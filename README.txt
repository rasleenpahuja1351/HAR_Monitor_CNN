HAR Monitor · Human Activity Recognition for Factory Automation

Group ID: EIC-16 · Course: UCS321 EST AI Project · Session: July–December 2026 Department of
Computer Science & Engineering, Thapar Institute of Engineering and Technology, Patiala

  Name                 Roll number
  -------------------- -------------
  Rasleen Pahuja       1025050106
  Bishal Barun Borah   1025050115
  Mahak Aggarwal       1025050117
  Vanshit Singla       1025050121
  Akshita Aggarwal     1025050129

Live website: https://har-monitor.vercel.app · API docs: /api/docs

----------------------------------------------------------------------------------------------------

1. What this project does

Factory floors are shared by people, robots and forklifts, but most machines cannot tell whether a
nearby worker is walking, climbing stairs, standing at a station or lying on the floor after a fall.
This project recognises six human activities from a waist-worn smartphone's accelerometer and
gyroscope:

LAYING · SITTING · STANDING · WALKING · WALKING_UPSTAIRS · WALKING_DOWNSTAIRS

Every 2.56-second window of sensor data is summarised as 561 features, classified by a neural
network, and turned into a line-safety response (for example "Operator on stairs: pause forklift
lane below"). Predictions under 70% confidence are sent for human review instead of triggering a
machine.

We trained and compared three models on the same person-wise split:

  Model                             Test accuracy   Precision   Recall   Macro F1   Validation acc.   Size
  --------------------------------- --------------- ----------- -------- ---------- ----------------- ----------------
  Dense neural network (deployed)   93.5%           0.942       0.934    0.935      97.1%             455,430 params
  Random Forest (baseline)          92.9%           0.930       0.926    0.927      93.1%             300 trees
  1D CNN                            89.1%           0.906       0.891    0.890      92.1%             53,222 params

Test set: 2,947 windows from 9 people the models never saw during training. Full per-class results
are in results/summary.md and results/metrics_all.json.

Key findings

-   Laying (a possible fall) is recognised almost perfectly by every model (F1 ≥ 0.978).
-   Sitting vs standing is the hardest pair: the Dense network's largest error is 103 sitting
    windows predicted as standing, because a waist-worn phone sees almost the same posture.
-   The 1D CNN is 8.5× smaller than the Dense network but less accurate, because the 561 inputs are
    summary statistics whose column order carries little local structure for convolution to exploit.

----------------------------------------------------------------------------------------------------

2. Dataset

UCI Human Activity Recognition Using Smartphones — Reyes-Ortiz, Anguita, Ghio, Oneto & Parra (2012).
UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/240 , DOI 10.24432/C54S4K. We
use the Kaggle CSV version (train.csv, test.csv).

              Windows   People   Columns
  ----------- --------- -------- -----------------------------------
  train.csv   7,352     21       561 features + subject + Activity
  test.csv    2,947     9        same

No missing values, no duplicates, all features in [−1, 1], classes roughly balanced. See
data/README.md for download instructions.

----------------------------------------------------------------------------------------------------

3. Method

1.  Person-wise split (GroupShuffleSplit on subject, seed 42): 16 people train (5,551 windows,
    53.9%), 5 people validation (1,801, 17.5%), 9 people test (2,947, 28.6%). No person appears in
    two sets, so scores reflect performance on a new worker.
2.  Preprocessing: LabelEncoder for the activity names, StandardScaler fitted on the training people
    only. No augmentation (balanced classes, pre-computed features).
3.  Models (definitions in src/models.py, hyperparameters in configs/config.json):
    -   Dense NN: 561 → 512 → 256 → 128 → 6; each hidden layer Dense + BatchNorm + ReLU + Dropout
        (0.4/0.4/0.3); softmax output.
    -   1D CNN: input 561×1 → 3 blocks of Conv1D (32/64/128 filters, kernels 7/5/3) + BatchNorm +
        ReLU + MaxPool → Global Average Pooling → Dense 128 + Dropout 0.4 → softmax.
    -   Random Forest: 300 trees.
4.  Training: Adam (lr 0.001), batch 64, sparse categorical cross-entropy, up to 200 epochs,
    EarlyStopping (patience 15, best weights restored), ReduceLROnPlateau (×0.5 after 5 flat
    epochs). Dense NN best epoch 14, CNN best epoch 33.
5.  Deployment: BatchNorm and the scaler are folded into the Dense weights (src/export_model.py), so
    the web API runs the network with NumPy only — no TensorFlow on the server. An automatic parity
    check confirms 100% identical predictions to Keras.

----------------------------------------------------------------------------------------------------

4. Project structure

    03_Model_Code/
    ├── app.py                  FastAPI backend (Vercel entry point) – hardened API
    ├── predict.py              Command-line prediction on a CSV (NumPy only)
    ├── run_experiments.py      Train / evaluate all three models, write results/
    ├── configs/config.json     Seed, paths, split and all hyperparameters
    ├── src/
    │   ├── data.py             Load UCI HAR, label encoding, person-wise split
    │   ├── models.py           Dense NN, 1D CNN, Random Forest definitions
    │   ├── train.py            Training + evaluation of each model
    │   ├── evaluate.py         Metrics (accuracy, precision, recall, F1) and figures
    │   ├── export_model.py     Keras → NumPy export with parity check
    │   ├── inference.py        NumPy inference used by the API and predict.py
    │   └── config.py           Config loader
    ├── checkpoints/            Trained models (.keras, .joblib) + training histories
    ├── model/                  Exported weights, metrics and live samples used by the website
    ├── results/                metrics_all.json, summary.md, figures/*.png, website/*.jpg (screenshots)
    ├── public/                 Website (HTML, CSS, JavaScript)
    ├── tests/test_api.py       18 automated API tests (normal use + failure cases)
    ├── data/                   Put train.csv and test.csv here (see data/README.md)
    ├── report/                 Project report (PDF)
    ├── viva_notes.md           Likely evaluation questions and answers
    ├── requirements.txt        Runtime dependencies (used by Vercel)
    ├── requirements-train.txt  Training dependencies (TensorFlow, scikit-learn, …)
    ├── requirements-dev.txt    Local server + tests
    ├── vercel.json, .vercelignore, .python-version
    └── README.md / README.txt

----------------------------------------------------------------------------------------------------

5. How to run

Python 3.12 is recommended (3.10+ works).

    # 1) Website + API locally
    pip install -r requirements-dev.txt
    uvicorn app:app --reload            # open http://127.0.0.1:8000  (API docs: /api/docs)
    pytest                              # 18 tests

    # 2) Predict a CSV from the command line (no TensorFlow needed)
    python predict.py data/test.csv --out results/predictions.csv

    # 3) Reproduce the results
    pip install -r requirements-train.txt
    python -m src.data data                          # dataset checks: shapes, missing values, split
    python run_experiments.py --evaluate-only        # recompute results/ from the saved checkpoints (~15 s)
    python run_experiments.py                        # retrain all three models from scratch (CPU: several minutes)
    python run_experiments.py --export               # retrain and refresh model/ for the website
    python run_experiments.py --quick --out /tmp/x   # 2-epoch smoke test that does not touch results/

Retraining on a different machine can change the last digit of the scores slightly (floating-point
and thread scheduling); --evaluate-only reproduces the reported numbers exactly.

----------------------------------------------------------------------------------------------------

6. Deploy on Vercel

This folder is a complete Vercel project (FastAPI is detected automatically from app.py; public/ is
served by the CDN).

Option A — GitHub (recommended)

1.  Create a GitHub repository and upload the contents of this 03_Model_Code folder to its root (so
    app.py is at the top level).
2.  In Vercel: Add New → Project → Import the repository. Leave the framework preset and build
    settings on their defaults and click Deploy.
3.  To update an existing Vercel project (e.g. har-monitor), replace the repository's files with
    these and push; Vercel redeploys automatically.

If you upload the whole EIC-16 folder instead, open Project Settings → General → Root Directory and
set it to 03_Model_Code.

Option B — Vercel CLI

    npm i -g vercel
    cd 03_Model_Code
    vercel            # preview deployment
    vercel --prod     # production

After deploying, check https://<your-app>.vercel.app/api/health → {"status": "ok", ...}.

----------------------------------------------------------------------------------------------------

7. API

  Method   Route                Purpose
  -------- -------------------- ------------------------------------------------------------------------
  GET      /api/health          Status, version, classes (503 degraded if the model files are missing)
  GET      /api/metrics         Accuracy, F1, confusion matrix, training curves (cached)
  GET      /api/features        The 561 feature names in the expected order
  POST     /api/predict         {"features": [561 numbers]} → activity, confidence, probabilities
  POST     /api/predict/batch   {"rows": [[561 numbers], …]} (1–1000 rows) → predictions + confidences
  GET      /api/stream?n=1      Classifies real held-out test windows (live demo); n 1–60

Why it does not crash

-   Strict validation: exactly 561 finite numbers per row, |x| ≤ 10⁴, no NaN/Infinity, no unknown
    fields, 1–1000 rows per batch → clear 422 messages.
-   Request bodies over 4 MB are refused with 413 before parsing (Vercel's own limit is 4.5 MB); the
    website uploads large CSVs in 250-row chunks.
-   Model, metrics and samples load lazily and are cached; if a file is missing the API answers
    503 model_unavailable instead of crashing.
-   Every unexpected error becomes a JSON 500 with a request ID; stack traces are never exposed.
-   Consistent error format {"detail": "...", "error": "<code>"}, plus X-Request-ID, security and
    cache headers on every response.
-   The website retries failed requests with back-off and time-outs, never overlaps stream requests,
    pauses the live stream after repeated failures with a Reconnect button, and shows a message
    instead of empty charts.
-   tests/test_api.py covers bad JSON, wrong lengths, NaN, huge values, oversized bodies, unknown
    routes, missing model files and simulated internal errors.

----------------------------------------------------------------------------------------------------

8. Own work and external sources

  Component                                                                                 Source
  ----------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------
  Dataset and its 561 features                                                              External: UCI Machine Learning Repository (Reyes-Ortiz et al., 2012), via Kaggle
  Libraries: TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, FastAPI, Pydantic   External open-source libraries
  Data loading, person-wise split, training/evaluation scripts, model definitions           Written for this project
  Keras → NumPy export with BatchNorm folding, FastAPI backend, website, tests              Written for this project
  All reported results and figures                                                          Produced by our own trained models
  AI assistance                                                                             An AI assistant (Claude) helped draft parts of the code, website and documentation; the group reviewed, ran and verified all of it

Fonts on the website: Instrument Serif, Inter and JetBrains Mono (Google Fonts, SIL Open Font
License).

9. References

1.  Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2012). Human Activity
    Recognition Using Smartphones [Dataset]. UCI Machine Learning Repository.
    https://doi.org/10.24432/C54S4K
2.  Anguita, D., et al. (2013). A public domain dataset for human activity recognition using
    smartphones. ESANN 2013.
3.  Ronao, C. A., & Cho, S.-B. (2016). Human activity recognition with smartphone sensors using deep
    learning neural networks. Expert Systems with Applications, 59, 235–244.
    https://doi.org/10.1016/j.eswa.2016.04.032
4.  Ordóñez, F. J., & Roggen, D. (2016). Deep convolutional and LSTM recurrent neural networks for
    multimodal wearable activity recognition. Sensors, 16(1), 115. https://doi.org/10.3390/s16010115
5.  Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.
    https://doi.org/10.1023/A:1010933404324
6.  Ioffe, S., & Szegedy, C. (2015). Batch normalization. ICML 2015.
    https://arxiv.org/abs/1502.03167
7.  Keras (https://keras.io), scikit-learn (https://scikit-learn.org), FastAPI
    (https://fastapi.tiangolo.com), Vercel Python runtime
    (https://vercel.com/docs/functions/runtimes/python).
