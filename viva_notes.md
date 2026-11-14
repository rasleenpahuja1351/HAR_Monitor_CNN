# Viva notes · Group EIC-16

Short answers to questions evaluators are likely to ask. Every member should be able to answer all of them.

## Problem and data

**1. What problem are you solving?**
Recognising what a factory worker is doing (laying, sitting, standing, walking, walking upstairs, walking downstairs) from a waist-worn phone's motion sensors, so machines can react: slow a robot cell, pause a forklift lane, or alert a supervisor about a possible fall.

**2. Which dataset, and where is it from?**
UCI Human Activity Recognition Using Smartphones (Reyes-Ortiz et al., 2012, DOI 10.24432/C54S4K), Kaggle CSV version. 30 volunteers aged 19–48 wore a Samsung Galaxy S II on the waist. 10,299 windows: 7,352 train (21 people) and 2,947 test (9 people).

**3. What is one sample?**
One 2.56-second window (128 readings at 50 Hz, 50% overlap with the next) summarised as 561 numbers: means, standard deviations, energy, entropy, correlations and FFT statistics of body acceleration, gravity and gyroscope signals.

**4. Did you extract the 561 features yourselves?**
No. The dataset authors computed them; we say so in the report. We used all 561 so every model sees identical inputs.

**5. Any missing values or class imbalance?**
No missing values or duplicates. Classes are roughly balanced (smallest is 70% of the largest), so no class weights or augmentation were needed.

## Preprocessing and evaluation

**6. How did you split the data, and why that way?**
By person. The 9 test people are only used once at the end. The 21 training people are split with GroupShuffleSplit into 16 for training and 5 for validation. A random row split would put the same person in train and test and inflate the accuracy.

**7. Why standardise when the features are already in [−1, 1]?**
Zero mean and unit variance help gradient descent converge. The scaler is fitted on the 16 training people only, to avoid leakage.

**8. Why is validation accuracy (97.1%) higher than test accuracy (93.5%)?**
The validation set has only 5 people, who happened to be easier to recognise. The 9-person test set is the honest estimate.

**9. Why report precision, recall and F1, not just accuracy?**
Accuracy hides which classes fail. For example, the CNN's standing recall is only 0.701 while laying is 0.970. Macro F1 gives every activity equal weight.

## Models

**10. Explain the 1D CNN.**
The 561 features are treated as a sequence of length 561 with one channel. Three convolution blocks (32/64/128 filters, kernel sizes 7/5/3, each with BatchNorm, ReLU and max-pooling) learn local patterns. Global average pooling, a 128-unit Dense layer with dropout and a softmax output give six probabilities. It has 53,222 parameters.

**11. Explain the Dense network.**
561 → 512 → 256 → 128 → 6. Each hidden layer is Dense, BatchNorm, ReLU and Dropout (0.4/0.4/0.3), with a softmax output. It has 455,430 parameters.

**12. Why did the CNN do worse than the Dense network?**
Convolution assumes that neighbouring inputs are related, which is true for raw signals. Our inputs are summary statistics whose column order carries little meaning. The Dense layers and the Random Forest can combine any features freely. The CNN's validation accuracy was also unstable (36–92% between epochs).

**13. What do BatchNorm, Dropout, EarlyStopping and ReduceLROnPlateau do?**
- BatchNorm normalises activations so training is stable.
- Dropout randomly switches off neurons to reduce overfitting.
- EarlyStopping stops when validation loss stops improving (patience 15) and restores the best weights.
- ReduceLROnPlateau halves the learning rate after 5 epochs without improvement.

**14. Loss, optimiser and hyperparameters?**
Sparse categorical cross-entropy, Adam with learning rate 0.001, batch size 64, up to 200 epochs, seed 42. All values are in `configs/config.json`.

**15. Why a Random Forest baseline?**
It is the strongest classical model for tabular features and needs little tuning. It scored 92.9%, only 0.6 points behind the Dense network and best on laying, sitting and standing.

## Results

**16. Headline results?**
On the 9 unseen test people:

| Model | Accuracy | Macro F1 |
|---|---|---|
| Dense NN | 93.5% | 0.935 |
| Random Forest | 92.9% | 0.927 |
| 1D CNN | 89.1% | 0.890 |

Laying is recognised almost perfectly by all three.

**17. Biggest error and why?**
The Dense network predicts 103 sitting windows as standing. A waist-worn phone barely changes orientation between those two still postures. Neither is a safety risk.

## Deployment

**18. How is the model deployed?**
FastAPI on Vercel. TensorFlow is too large for a serverless function, so we folded BatchNorm and the scaler into the Dense weights (W′ = W·k, b′ = (b − μ)·k + β) and run inference with NumPy. A parity check shows 100% identical predictions to Keras.

**19. Why does the website not crash on bad input?**
- **Validation:** inputs are strictly checked (561 finite values, bounded range, row limits), and invalid ones get a clear 422 error.
- **Large requests:** bodies over 4 MB are refused with 413.
- **Missing model files:** the API returns 503 instead of crashing.
- **Unexpected errors:** they become a JSON 500 with a request ID.
- **Front end:** it retries failed requests with timeouts and pauses the live stream with a Reconnect button.
- **Tests:** 18 automated tests cover these cases.

**20. What does the live stream show, and is it real data?**
Yes, it uses real held-out test windows (100 per class) classified by the deployed model every 2.56 s. Each prediction is shown with its true label, confidence and a factory response. Predictions below 70% confidence go to human review.

## Limitations and future work

**21. Limitations?**
- The data was recorded in a controlled setting by 30 volunteers, not by real factory workers.
- There is only one sensor position.
- Sitting and standing are still confused.
- The CNN was given pre-computed features rather than raw signals.

**22. What would you do next?**
- Train the CNN or a CNN-LSTM on the raw 128 × 9 signals.
- Use subject-wise k-fold cross-validation.
- Ensemble the three models.
- Collect data from factory workers, including lifting and pushing.
- Run the model on the wearable itself.

## Who did what (fill in before the viva)

| Member | Main contribution |
|---|---|
| Rasleen Pahuja (1025050106) | |
| Bishal Barun Borah (1025050115) | |
| Mahak Aggarwal (1025050117) | |
| Vanshit Singla (1025050121) | |
| Akshita Aggarwal (1025050129) | |
