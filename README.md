# Project 3 — Cancer Diagnosis (Spark MLlib)

This project trains and compares Decision Tree vs Random Forest classifiers on the provided breast cancer dataset using PySpark MLlib.

## Dataset
- File: `project3_data.csv`
- Label column: `diagnosis` (`B` = benign, `M` = malignant)
- Feature columns: all other numeric columns (we drop `id`)

## Requirements
- Apache Spark (I tested with Spark 3.5.1)
- Python + numpy

### Conda setup (that I used)

```bash
conda create -y -n project3-sparkml python=3.11 numpy
```

## Run
Run from the project directory:

```bash
PY="$(conda run -n project3-sparkml python -c 'import sys; print(sys.executable)')"
export PYSPARK_PYTHON="$PY"
export PYSPARK_DRIVER_PYTHON="$PY"

spark-submit \
  --files log4j2.properties \
  --conf "spark.driver.extraJavaOptions=-Dlog4j2.configurationFile=log4j2.properties" \
  --conf "spark.executor.extraJavaOptions=-Dlog4j2.configurationFile=log4j2.properties" \
  src/train_models.py \
  --data "project3_data.csv" \
  --out results.json \
  --log-level WARN
```

### Note
- I used 80/20 data split for a better estimate but you can change it by adding `--train-frac 0.7`


## Outputs
- results.json: metrics + confusion matrix for both models (Decision Tree and Random Forest)
- This file generates 

Metrics included:
- accuracy
- precision / recall / F1 (weighted + per-class)
- confusion matrix (rows = true label, cols = predicted label)