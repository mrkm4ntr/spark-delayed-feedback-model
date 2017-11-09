package org.apache.spark.ml.classification

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, LBFGS => BreezeLBFGS, OWLQN => BreezeOWLQN}
import dfm.feature.Instance
import dfm.optim.aggregator.DFMAggregator
import dfm.param.HasTimeCol
import org.apache.spark.SparkException
import org.apache.spark.ml.linalg.{BLAS, DenseMatrix, Matrices, Matrix, Vector, Vectors}
import org.apache.spark.ml.optim.loss.{L2Regularization, RDDLossFunction}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{Identifiable, Instrumentation}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable

trait DelayedFeedbackClassifierParam extends ProbabilisticClassifierParams
  with HasRegParam with HasElasticNetParam with HasMaxIter with HasFitIntercept with HasTol
  with HasStandardization with HasWeightCol with HasThreshold with HasAggregationDepth with HasTimeCol

class DelayedFeedbackClassifier(
    override val uid: String)
  extends ProbabilisticClassifier[Vector, DelayedFeedbackClassifier, DelayedFeedbackClassifierModel]
  with DelayedFeedbackClassifierParam {

  def this() = this(Identifiable.randomUID("dfc"))

  def setRegParam(value: Double): this.type  = set(regParam, value)
  setDefault(regParam -> 0.0)

  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)
  setDefault(elasticNetParam -> 0.0)

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
  setDefault(fitIntercept -> true)

  def setStandardization(value: Boolean): this.type = set(standardization, value)
  setDefault(standardization -> true)

  def setThreshold(value: Double): this.type = set(threshold, value)
  setDefault(threshold -> 0.5)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)
  setDefault(aggregationDepth -> 2)

  def setTimeCol(value: String): this.type = set(timeCol, value)

  override def copy(extra: ParamMap): DelayedFeedbackClassifier = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): DelayedFeedbackClassifierModel = {
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val t = col($(timeCol))
    val instances: RDD[Instance] =
      dataset.select(col($(labelCol)), w, t, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, time: Double, features: Vector) =>
          Instance(label, weight, time, features)
      }

    val handlePersistence = dataset.storageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val instr = Instrumentation.create(this, instances)
    // TODO: Add
    instr.logParams(maxIter)

    val (summarizer, labelSummarizer) = {
      val seqOp = (c: (MultivariateOnlineSummarizer, DFMSummarizer),
        instance: Instance) =>
        (c._1.add(instance.features, instance.weight), c._2.add(instance.label, instance.weight))

      val combOp = (c1: (MultivariateOnlineSummarizer, DFMSummarizer),
        c2: (MultivariateOnlineSummarizer, DFMSummarizer)) =>
        (c1._1.merge(c2._1), c1._2.merge(c2._2))

      instances.treeAggregate(
        (new MultivariateOnlineSummarizer, new DFMSummarizer)
      )(seqOp, combOp, $(aggregationDepth))
    }

    val histogram = labelSummarizer.histogram
    val numInvalid = labelSummarizer.countInvalid
    val numFeatures = summarizer.mean.size
    // TODO: getFitIntercept
    val numFeaturesPlusIntercept = if ($(fitIntercept)) numFeatures + 1 else numFeatures

    instr.logNumClasses(2)
    instr.logNumFeatures(numFeatures)

    val (coefficientMatrix, interceptVector, objectiveHistory) = {
      if (numInvalid != 0) {
        // TODO: change message
        val msg = s"Classification labels should be in [0 to ${1}]. " +
          s"Found $numInvalid invalid labels."
        logError(msg)
        throw new SparkException(msg)
      }

      val featuresMean = summarizer.mean.toArray
      val featuresStd = summarizer.variance.toArray.map(math.sqrt)

      if (!$(fitIntercept) && (0 until numFeatures).exists { i =>
        featuresStd(i) == 0.0 && featuresMean(i) != 0.0 }) {
        logWarning("Fitting DelayedFeedbackClassifierModel without intercept on dataset with " +
          "constant nonzero column, Spark MLlib outputs zero coefficients for constant " +
          "nonzero columns. This behavior is the same as R glmnet but different from LIBSVM.")
      }

      val regParamL1 = $(elasticNetParam) * $(regParam)
      val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)

      val bcFeaturesStd = instances.context.broadcast(featuresStd)
      val getAggregatorFunc = new DFMAggregator(bcFeaturesStd, $(fitIntercept))(_)

      val regularization = if (regParamL2 != 0.0)
        Some(new L2Regularization(
          regParamL2,
          (idx: Int) => idx >= 0 && idx < numFeatures || idx >= numFeaturesPlusIntercept && idx < numFeaturesPlusIntercept + numFeatures,
          if ($(standardization)) None else Some((j: Int) => if (j >= 0 && j < numFeatures || j >= numFeaturesPlusIntercept && j < numFeaturesPlusIntercept + numFeatures) {
            featuresStd(j)
          } else 0.0)
        ))
      else None

      val costFun = new RDDLossFunction(instances, getAggregatorFunc, regularization, $(aggregationDepth))

      val optimizer = if ($(elasticNetParam) == 0.0 || $(regParam) == 0.0) {
        new BreezeLBFGS[BDV[Double]]($(maxIter), 10, $(tol))
      } else {
        val standardizationParam = $(standardization)
        def regParamL1Fun = (index: Int) => {
          // Remove the L1 penalization on the intercept
          val isIntercept = $(fitIntercept) && index >= numFeatures * 2
          if (isIntercept) {
            0.0
          } else {
            if (standardizationParam) {
              regParamL1
            } else {
              val featureIndex = index % numFeatures
              // If `standardization` is false, we still standardize the data
              // to improve the rate of convergence; as a result, we have to
              // perform this reverse standardization by penalizing each component
              // differently to get effectively the same objective function when
              // the training dataset is not standardized.
              if (featuresStd(featureIndex) != 0.0) {
                regParamL1 / featuresStd(featureIndex)
              } else {
                0.0
              }
            }
          }
        }
        new BreezeOWLQN[Int, BDV[Double]]($(maxIter), 10, regParamL1Fun, $(tol))
      }

      val initialCoefWithInterceptMatrix =
        Matrices.zeros(2, numFeaturesPlusIntercept)

      val states = optimizer.iterations(new CachedDiffFunction(costFun),
        new BDV[Double](initialCoefWithInterceptMatrix.toArray))

      val arrayBuilder = mutable.ArrayBuilder.make[Double]
      var state: optimizer.State = null
      while (states.hasNext) {
        state = states.next()
        arrayBuilder += state.adjustedValue
      }
      bcFeaturesStd.destroy(blocking = false)

      if (state == null) {
        val msg = s"${optimizer.getClass.getName} failed."
        logError(msg)
        throw new SparkException(msg)
      }

      val allCoefficients = state.x.toArray.clone()
      val allCoefMatrix = new DenseMatrix(numFeaturesPlusIntercept, 2,
        allCoefficients)
      val denseCoefficientMatrix = new DenseMatrix(2, numFeatures,
        new Array[Double](2 * numFeatures), isTransposed = true)
      val interceptVec = if ($(fitIntercept)) {
        Vectors.zeros(2)
      } else {
        Vectors.sparse(2, Seq.empty)
      }
      // separate intercepts and coefficients from the combined matrix
      allCoefMatrix.foreachActive { (featureIndex, groupIndex, value) =>
        val isIntercept = $(fitIntercept) && (featureIndex == numFeatures || featureIndex == numFeatures + numFeaturesPlusIntercept)
        if (!isIntercept && featuresStd(featureIndex) != 0.0) {
          denseCoefficientMatrix.update(groupIndex, featureIndex,
            value / featuresStd(featureIndex))
        }
        if (isIntercept) interceptVec.toArray(groupIndex) = value
      }
      (denseCoefficientMatrix.compressed, interceptVec.compressed, arrayBuilder.result())
    }

    if (handlePersistence) instances.unpersist()

    val model = copyValues(new DelayedFeedbackClassifierModel(uid, coefficientMatrix, interceptVector))
    model
  }
}

class DelayedFeedbackClassifierModel(
    override val uid: String,
    val coefficientMatrix: Matrix,
    val interceptVector: Vector)
  extends ProbabilisticClassificationModel[Vector, DelayedFeedbackClassifierModel]
  with DelayedFeedbackClassifierParam {

  require(coefficientMatrix.numRows == interceptVector.size, s"Dimension mismatch! Expected " +
    s"coefficientMatrix.numRows == interceptVector.size, but ${coefficientMatrix.numRows} != " +
    s"${interceptVector.size}")

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = rawPrediction

  override def numClasses: Int = 2

  override protected def predictRaw(features: Vector): Vector = {
    val (wc, wd) = margins(features)
    val p = 1.0 / (1.0 + math.exp(-wc))
    val l = math.exp(wd)
    val p2 = p * (1.0 - math.exp(-l))
    Vectors.dense(1.0 - p2, p2)
  }

  private val margins: Vector => (Double, Double) = (features) => {
    val m = interceptVector.toDense.copy
    BLAS.gemv(1.0, coefficientMatrix, features, 1.0, m)
    val values = m.values
    (values(0), values(1))
  }

  override def copy(extra: ParamMap): DelayedFeedbackClassifierModel = {
    val newModel = copyValues(new DelayedFeedbackClassifierModel(uid, coefficientMatrix, interceptVector), extra)
    newModel.setParent(parent)
  }
}

class DFMSummarizer extends Serializable {
  // The first element of value in distinctMap is the actually number of instances,
  // and the second element of value is sum of the weights.
  private val distinctMap = new mutable.HashMap[Int, (Long, Double)]
  private var totalInvalidCnt: Long = 0L

  /**
    * Add a new label into this MultilabelSummarizer, and update the distinct map.
    *
    * @param label The label for this data point.
    * @param weight The weight of this instances.
    * @return This MultilabelSummarizer
    */
  def add(label: Double, weight: Double = 1.0): this.type = {
    require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

    if (weight == 0.0) return this

    if (label - label.toInt != 0.0 || label < 0) {
      totalInvalidCnt += 1
      this
    }
    else {
      val (counts: Long, weightSum: Double) = distinctMap.getOrElse(label.toInt, (0L, 0.0))
      distinctMap.put(label.toInt, (counts + 1L, weightSum + weight))
      this
    }
  }

  /**
    * Merge another MultilabelSummarizer, and update the distinct map.
    * (Note that it will merge the smaller distinct map into the larger one using in-place
    * merging, so either `this` or `other` object will be modified and returned.)
    *
    * @param other The other MultilabelSummarizer to be merged.
    * @return Merged MultilabelSummarizer object.
    */
  def merge(other: DFMSummarizer): DFMSummarizer = {
    val (largeMap, smallMap) = if (this.distinctMap.size > other.distinctMap.size) {
      (this, other)
    } else {
      (other, this)
    }
    smallMap.distinctMap.foreach {
      case (key, value) =>
        val (counts: Long, weightSum: Double) = largeMap.distinctMap.getOrElse(key, (0L, 0.0))
        largeMap.distinctMap.put(key, (counts + value._1, weightSum + value._2))
    }
    largeMap.totalInvalidCnt += smallMap.totalInvalidCnt
    largeMap
  }

  /** @return The total invalid input counts. */
  def countInvalid: Long = totalInvalidCnt

  /** @return The number of distinct labels in the input dataset. */
  def numClasses: Int = if (distinctMap.isEmpty) 0 else distinctMap.keySet.max + 1

  /** @return The weightSum of each label in the input dataset. */
  def histogram: Array[Double] = {
    val result = Array.ofDim[Double](numClasses)
    var i = 0
    val len = result.length
    while (i < len) {
      result(i) = distinctMap.getOrElse(i, (0L, 0.0))._2
      i += 1
    }
    result
  }
}
