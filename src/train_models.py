#!/usr/bin/env python3
"""
Train and evaluate Decision Tree vs Random Forest on the breast cancer dataset
using Spark MLlib (DataFrame-based pyspark.ml API).

Outputs accuracy, precision, recall, F1, and confusion matrix for both models.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession

#This will hold everything we want from a model
@dataclass
class ModelResults:
    model_name: str
    best_params: Dict[str, Any]
    labels: List[str]  # index -> original label string
    confusion_matrix: List[List[int]]  # rows=true labels, cols=pred labels
    metrics: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spark MLlib DT vs RF on cancer dataset")
    p.add_argument(
        "--data",
        required=True,
        help='Path to input CSV',
    )
    p.add_argument(
        "--out",
        default="results.json",
        help='Where to write results JSON',
    )
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Train fraction for randomSplit",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--log-level",
        default="WARN",
        help='Spark log level (I set it to WARN for less noisy output)',
    )
    return p.parse_args()

#Im getting a spark session
def get_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

#reading everything from the csv file except the id column
def load_data(spark: SparkSession, path: str) -> Tuple[DataFrame, List[str]]:
    df = spark.read.csv(path, header=True, inferSchema=True)
    if "diagnosis" not in df.columns:
        raise ValueError('Expected a "diagnosis" column in the input CSV')

    # Drop non-feature identifier column if present.
    if "id" in df.columns:
        df = df.drop("id")

    feature_cols = [c for c in df.columns if c != "diagnosis"]
    if not feature_cols:
        raise ValueError("No feature columns found (columns besides diagnosis/id)")
    return df, feature_cols

#building pipeline for the model
def build_pipeline(feature_cols: List[str], classifier) -> Pipeline:
    indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    return Pipeline(stages=[indexer, assembler, classifier])


def compute_metrics(predictions: DataFrame, labels: List[str]) -> Tuple[Dict[str, Any], List[List[int]]]:
    """
    Computes accuracy, weighted precision/recall/F1, per-class precision/recall/F1,
    and confusion matrix.
    """
    # Using Spark's Evaluators for accuracy and f1 score
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    weighted_f1 = float(eval_f1.evaluate(predictions))
    accuracy = float(eval_acc.evaluate(predictions))

    # Using MulticlassMetrics to get per-class metrics and confusion matrix
    from pyspark.mllib.evaluation import MulticlassMetrics

    pl_rdd = predictions.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))
    mm = MulticlassMetrics(pl_rdd)

    # Confusion matrix: rows = true labels, columns = predicted labels
    cm = mm.confusionMatrix().toArray()
    cm_int = [[int(x) for x in row] for row in cm.tolist()]

    # For each label, we compute per-class precision, recall and f1 score
    per_class: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(labels):
        per_class[name] = {
            "precision": float(mm.precision(float(idx))),
            "recall": float(mm.recall(float(idx))),
            "f1": float(mm.fMeasure(float(idx), beta=1.0)),
        }

    # Creating a dictionary to store all the metrics
    metrics = {
        "accuracy": accuracy,
        "test_error": 1.0 - accuracy,
        "weighted_precision": float(mm.weightedPrecision),
        "weighted_recall": float(mm.weightedRecall),
        "weighted_f1": float(mm.weightedFMeasure(beta=1.0)),
        "weighted_f1_evaluator": weighted_f1,  # Spark evaluator (should match closely)
        "per_class": per_class,
    }
    return metrics, cm_int


def extract_best_params(model_name: str, pipeline_model) -> Dict[str, Any]:
    """
    Pull key hyperparameters from the trained classifier stage for reporting.
    """
    classifier_model = pipeline_model.stages[-1]
    out: Dict[str, Any] = {"model": model_name}

    # Common tree params
    for k in ["maxDepth", "maxBins", "impurity", "minInstancesPerNode", "minInfoGain"]:
        if classifier_model.hasParam(k):
            out[k] = classifier_model.getOrDefault(k)

    # RF-specific
    for k in ["numTrees", "featureSubsetStrategy", "subsamplingRate"]:
        if classifier_model.hasParam(k):
            out[k] = classifier_model.getOrDefault(k)

    return out

#Building pipeline, using F1 score as the evaluator
def fit_and_eval(
    train_df: DataFrame,
    test_df: DataFrame,
    feature_cols: List[str],
    *,
    model_name: str,
    classifier,
    do_cv: bool,
    seed: int,
) -> ModelResults:
    pipeline = build_pipeline(feature_cols, classifier)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    if do_cv: #if true, run 5-fold cross-validation and choose the best model by F1 score
        from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

        grid = ParamGridBuilder()

        # Keep grids intentionally small (dataset is tiny; runtime matters on laptops).
        if model_name == "DecisionTree":
            grid = (
                grid.addGrid(classifier.maxDepth, [2, 4, 6])
                .addGrid(classifier.impurity, ["gini", "entropy"])
            )
        elif model_name == "RandomForest":
            grid = (
                grid.addGrid(classifier.numTrees, [20, 50])
                .addGrid(classifier.maxDepth, [4, 6])
                .addGrid(classifier.featureSubsetStrategy, ["auto", "sqrt"])
                .addGrid(classifier.impurity, ["gini", "entropy"])
            )

        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=grid.build(),
            evaluator=evaluator,
            numFolds=5,
            seed=seed,
            parallelism=2,
        )
        cv_model = cv.fit(train_df)
        best_model = cv_model.bestModel
    else:
        best_model = pipeline.fit(train_df)

    # Extracting labels list so we know what index corresponds to B or M
    indexer_model = best_model.stages[0]
    labels = list(indexer_model.labels)

    preds = best_model.transform(test_df).cache()
    metrics, confusion = compute_metrics(preds, labels)
    best_params = extract_best_params(model_name, best_model)

    return ModelResults(
        model_name=model_name,
        best_params=best_params,
        labels=labels,
        confusion_matrix=confusion,
        metrics=metrics,
    )


def main() -> None:
    args = parse_args()
    spark = get_spark("Project3-Cancer-MLlib")
    spark.sparkContext.setLogLevel(args.log_level.upper())

    df, feature_cols = load_data(spark, args.data)

    # Train/test split
    train_frac = args.train_frac
    if not (0.0 < train_frac < 1.0):
        raise ValueError("--train-frac must be between 0 and 1")
    train_df, test_df = df.randomSplit([train_frac, 1.0 - train_frac], seed=args.seed)

    # Cache for speed
    train_df = train_df.cache()
    test_df = test_df.cache()
    _ = train_df.count()
    _ = test_df.count()

    # Always run cross-validation
    do_cv = True

    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", featureSubsetStrategy="auto")

    dt_res = fit_and_eval(
        train_df,
        test_df,
        feature_cols,
        model_name="DecisionTree",
        classifier=dt,
        do_cv=do_cv,
        seed=args.seed,
    )
    rf_res = fit_and_eval(
        train_df,
        test_df,
        feature_cols,
        model_name="RandomForest",
        classifier=rf,
        do_cv=do_cv,
        seed=args.seed,
    )

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": args.data,
        "train_frac": train_frac,
        "seed": args.seed,
        "cross_validation": do_cv,
        "models": [asdict(dt_res), asdict(rf_res)],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=False)

    # Also print a short human-readable summary
    def _print_model(m: ModelResults) -> None:
        print(f"\n== {m.model_name} ==")
        print("Best params:", m.best_params)
        print("Labels (index -> label):", m.labels)
        print("Accuracy:", m.metrics["accuracy"])
        print("Precision (weighted):", m.metrics["weighted_precision"])
        print("Recall (weighted):", m.metrics["weighted_recall"])
        print("F1 (weighted):", m.metrics["weighted_f1"])
        print("Confusion matrix (rows=true, cols=pred):")
        for row in m.confusion_matrix:
            print(row)

    _print_model(dt_res)
    _print_model(rf_res)

    print(f"\nWrote results to: {args.out}")

    spark.stop()


if __name__ == "__main__":
    main()


