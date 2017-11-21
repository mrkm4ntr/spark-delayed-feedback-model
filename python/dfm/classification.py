from pyspark import keyword_only
from pyspark.ml.param.shared import *
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaParams
from pyspark.ml.util import JavaPredictionModel


class DelayedFeedbackClassifier(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasMaxIter,
                                HasRegParam, HasTol, HasProbabilityCol, HasRawPredictionCol,
                                HasElasticNetParam, HasFitIntercept, HasStandardization,
                                HasWeightCol, HasAggregationDepth):

    threshold = Param(Params._dummy(), "threshold",
                      "Threshold in binary classification prediction, in range [0, 1]." +
                      " If threshold and thresholds are both set, they must match." +
                      "e.g. if threshold is p, then thresholds must be equal to [1-p, p].",
                      typeConverter=TypeConverters.toFloat)

    timeCol = Param(Params._dummy(), "timeCol", "", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, featuresCol="features", lableCol="label", predictionCol="prediction",
                 maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True,
                 threshold=0.5, probabilityCol="probability",
                 rawPredictionCol="rawPrediction", standardization=True, weightCol=None,
                 aggregationDepth=2, timeCol=None):
        super(DelayedFeedbackClassifier, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.classification.DelayedFeedbackClassifier", self.uid)
        self._setDefault(maxIter=100, regParam=0.0, tol=1E-6, threshold=0.5)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _create_model(self, java_model):
        return DelayedFeedbackClassifierModel(java_model)


class DelayedFeedbackClassifierModel(JavaModel, JavaPredictionModel):

    @property
    def coefficientMatrix(self):
        return self._call_java("coefficientMatrix")

    @property
    def interceptVector(self):
        return self._call_java("interceptVector")
