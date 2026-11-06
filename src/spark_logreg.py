"""Logistic regression in Spark two ways on the Wisconsin breast-cancer data (569 rows, 30 features):

  1. RDD API  : pyspark.mllib LogisticRegressionWithSGD on LabeledPoint, raw (unscaled) features, 75/25 split
  2. DataFrame: pyspark.ml Pipeline (VectorAssembler -> StandardScaler -> LogisticRegression, L-BFGS), 80/20 split

Both use seed 42. The data comes from scikit-learn's bundled copy, so no download is needed.
Optional controlled run (--controlled): same 80/20 split and standardised features for both APIs.
"""
import json
import sys

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.datasets import load_breast_cancer

spark = (SparkSession.builder.master("local[2]").appName("LogRegRDDvsDF")
         .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "4").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

pdf = load_breast_cancer(as_frame=True).frame
pdf["target"] = pdf["target"].astype(float)
df = spark.createDataFrame(pdf)
features = [c for c in df.columns if c != "target"]


def rdd_sgd(split, scale=False):
    rdf = df
    if scale:
        pipe = Pipeline(stages=[VectorAssembler(inputCols=features, outputCol="raw"),
                                StandardScaler(inputCol="raw", outputCol="f", withMean=True, withStd=True)])
        rdf = pipe.fit(df).transform(df).select("f", "target")
        rdd = rdf.rdd.map(lambda r: LabeledPoint(r["target"], Vectors.dense(r["f"].toArray())))
    else:
        rdd = df.rdd.map(lambda r: LabeledPoint(r["target"], Vectors.dense([float(r[c]) for c in features])))
    train, test = rdd.randomSplit(split, seed=42)
    model = LogisticRegressionWithSGD.train(train, iterations=200, step=0.1)
    acc = lambda d: d.map(lambda lp: (float(model.predict(lp.features)), lp.label)).filter(lambda p: p[0] == p[1]).count() / d.count()
    return round(acc(train), 4), round(acc(test), 4)


def dataframe_lbfgs(split):
    train, test = df.randomSplit(split, seed=42)
    pipe = Pipeline(stages=[VectorAssembler(inputCols=features, outputCol="raw"),
                            StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True),
                            LogisticRegression(featuresCol="features", labelCol="target", maxIter=200)])
    model = pipe.fit(train)
    ev = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
    return round(ev.evaluate(model.transform(train)), 4), round(ev.evaluate(model.transform(test)), 4)


res = {
    "as_in_assignment": {
        "RDD SGD, raw features, 75/25": dict(zip(("train", "test"), rdd_sgd([0.75, 0.25]))),
        "DataFrame L-BFGS, standardised, 80/20": dict(zip(("train", "test"), dataframe_lbfgs([0.8, 0.2]))),
    }
}
if "--controlled" in sys.argv:
    res["controlled_80_20"] = {
        "RDD SGD, raw features": dict(zip(("train", "test"), rdd_sgd([0.8, 0.2]))),
        "RDD SGD, standardised": dict(zip(("train", "test"), rdd_sgd([0.8, 0.2], scale=True))),
        "DataFrame L-BFGS, standardised": dict(zip(("train", "test"), dataframe_lbfgs([0.8, 0.2]))),
    }
print(json.dumps(res, indent=2))
json.dump(res, open("results/logreg_results.json", "w"), indent=2)
spark.stop()
