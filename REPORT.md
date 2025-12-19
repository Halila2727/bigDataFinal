# Project III Report — Cancer Diagnosis using Spark MLlib

## 1. Problem statement
We build machine learning models to classify breast tumors as benign or malignant using 20+ clinical variables, and compare the performance of two algorithms using Spark MLlib.

## 2. Dataset
- Source file: `project3_data.csv`
- Samples: 569
- Label: `diagnosis` is {`B`, `M`}
- Features: all numeric columns except `id` (identifier)

## 3. Algorithms (justification)
We compare:

### 3.1 Decision Tree
- It is a good interpretable baseline model as it requires minimal assumptions about data distribution. It can capture non-linear feature interactions and there is minimal preprocessing required. If a tumor is malignant or benign, a decision tree can show its reasoning as to why it determined something was that, which is good to see its line of thinking.

### 3.2 Random Forest
- One of the biggest problems with a decision tree is that it can be prone to over-fitting. By Bagging, Random Forest addresses this. Also, Random Forest is considered in real life for tabular datasets like the one we have. As an ensemble of trees, it should reduce variance and improve generalization over a single tree. So, I think these two algorithms will show the trade-off between the high interpretability but lower accuracy of a decision tree and lower interpretability but higher accuracy of a random forest.

## 4. Training procedure
### 4.1 Preprocessing (Spark MLlib pipeline)
- Drop `id`
- Convert `diagnosis` to numeric `label` using `StringIndexer`
- Combine feature columns into a single vector using `VectorAssembler`

### 4.2 Train/test split
- Split method: `randomSplit([train_frac, 1-train_frac], seed=42)`
- Train fraction used: **0.8** (80/20)

### 4.3 Hyperparameter selection / cross-validation
Cross-validation is used to select hyperparameters:
- Method: 5-fold cross-validation
- Selection metric: F1 score

Hyperparameter search grid:
- Decision Tree: maxDepth ∈ {2,4,6}, impurity ∈ {gini, entropy}
- Random Forest: numTrees ∈ {20,50}, maxDepth ∈ {4,6}, featureSubsetStrategy ∈ {auto, sqrt}, impurity ∈ {gini, entropy}

## 5. Testing results

### 5.1 Metrics definitions
- **Precision**: \( TP / (TP + FP) \)
- **Recall**: \( TP / (TP + FN) \)
- **F1**: \( 2 \\cdot (Precision \\cdot Recall) / (Precision + Recall) \)
- **Accuracy**: \( \\#correct / \\#samples \)

### 5.2 Results table (test set)

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|------|----------|----------------------|-------------------|---------------|
| Decision Tree | 0.9419 | 0.9417 | 0.9419 | 0.9417 |
| Random Forest | 0.9651 | 0.9652 | 0.9651 | 0.9650 |

### 5.3 Confusion matrices
Use the confusion matrices reported in `results.json` (rows=true label, cols=predicted label).

Decision Tree (labels order: B, M):

```
[[52, 2],
 [3, 29]]
```

Random Forest (labels order: B, M):

```
[[53, 1],
 [2, 30]]
```

### 5.4 Per-class metrics

Per-class metrics (test set):
- Decision Tree:
  - B: precision=0.9455, recall=0.9630, F1=0.9541
  - M: precision=0.9355, recall=0.9063, F1=0.9206
- Random Forest:
  - B: precision=0.9636, recall=0.9815, F1=0.9725
  - M: precision=0.9677, recall=0.9375, F1=0.9524

## 6. Comparison and discussion
- As can be deduced clearly from the numbers above, Random Forest performed better than the single Decision Tree on the test set for every reported metric:
  - Accuracy improved from 0.9419 to 0.9651.
  - Weighted F1 improved from 0.9417 to 0.9650.
So, the ensemble approach of the Random Forest has reduced variance and error inherent in a single Decision Tree.

- In terms of the confusion matrix, Random Forest made fewer mistakes and also reduced the most important error type for cancer screening (predicting benign when it is actually malignant):
  - Decision Tree errors: FP=2 (benign predicted as malignant) and FN=3 (malignant predicted as benign).
  - Random Forest errors: FP=1 and FN=2.

- Recall for malignant cases matters because it measures how many true malignant tumors the model correctly flags as malignant. A false negative (malignant predicted as benign) can delay diagnosis and treatment which is a huge problem. Here, Random Forest had higher malignant recall:
  - Decision Tree `M` recall: 0.9063
  - Random Forest `M` recall: 0.9375

- This result matches what I predicted about the trade-off between these two algorithms. A single Decision Tree is faster, but can have high variance as it is more sensitive to the training data. Random Forest reduces variance by averaging many trees (bagging), which improves generalization. As a result, I think we can safely say that the more complex Random Forest model is the way to go.

## 7. Limitations and future improvements
- The dataset size was relatively small, so the actual number in terms of differences were very small.
- I also used one seed. I could use repeated runs with different seeds and report mean±std.
- I could definitely try other algorithms discussed in class such as  (Logistic Regression, Linear SVM, Gradient-Boosted Trees) for broader comparison.
- Random Forest can provide a measure of feature importance, which estimates which clinical variables contributed most to the model’s decisions. This would help explain why the model predicts malignant vs benign (e.g., which “mean” or “worst” measurements are most influential) and could also help with feature selection by removing low-importance features to simplify the model. But I have not implemented that here.

## 8. My results
The results are stored in `results.json`, which is also provided.