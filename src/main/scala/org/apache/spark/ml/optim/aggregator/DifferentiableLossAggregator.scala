package org.apache.spark.ml.optim.aggregator

import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}

/**
  * A parent trait for aggregators used in fitting MLlib models. This parent trait implements
  * some of the common code shared between concrete instances of aggregators. Subclasses of this
  * aggregator need only implement the `add` method.
  *
  * @tparam Datum The type of the instances added to the aggregator to update the loss and gradient.
  * @tparam Agg Specialization of [[DifferentiableLossAggregator]]. Classes that subclass this
  *             type need to use this parameter to specify the concrete type of the aggregator.
  */
trait DifferentiableLossAggregator[
    Datum,
    Agg <: DifferentiableLossAggregator[Datum, Agg]] extends Serializable {

  self: Agg => // enforce classes that extend this to be the same type as `Agg`

  protected var weightSum: Double = 0.0
  protected var lossSum: Double = 0.0

  /** The dimension of the gradient array. */
  protected val dim: Int

  /** Array of gradient values that are mutated when new instances are added to the aggregator. */
  protected lazy val gradientSumArray: Array[Double] = Array.ofDim[Double](dim)

  /** Add a single data point to this aggregator. */
  def add(instance: Datum): Agg

  /** Merge two aggregators. The `this` object will be modified in place and returned. */
  def merge(other: Agg): Agg = {
    require(dim == other.dim, s"Dimensions mismatch when merging with another " +
      s"${getClass.getSimpleName}. Expecting $dim but got ${other.dim}.")

    if (other.weightSum != 0) {
      weightSum += other.weightSum
      lossSum += other.lossSum

      var i = 0
      val localThisGradientSumArray = this.gradientSumArray
      val localOtherGradientSumArray = other.gradientSumArray
      while (i < dim) {
        localThisGradientSumArray(i) += localOtherGradientSumArray(i)
        i += 1
      }
    }
    this
  }

  /** The current weighted averaged gradient. */
  def gradient: Vector = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but was $weightSum.")
    val result = Vectors.dense(gradientSumArray.clone())
    BLAS.scal(1.0 / weightSum, result)
    result
  }

  /** Weighted count of instances in this aggregator. */
  def weight: Double = weightSum

  /** The current loss value of this aggregator. */
  def loss: Double = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but was $weightSum.")
    lossSum / weightSum
  }

}
