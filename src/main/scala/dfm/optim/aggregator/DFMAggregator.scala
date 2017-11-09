package dfm.optim.aggregator

import breeze.numerics.sigmoid
import dfm.feature.Instance
import dfm.util.MLUtils
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.optim.aggregator.DifferentiableLossAggregator

class DFMAggregator(
    bcFeaturesStd: Broadcast[Array[Double]],
    fitIntercept: Boolean)(bcCoefficients: Broadcast[Vector])
  extends DifferentiableLossAggregator[Instance, DFMAggregator] with Logging {

  private val numFeatures = bcFeaturesStd.value.length
  private val numFeaturesPlusIntercept = if (fitIntercept) numFeatures + 1 else numFeatures
  private val coefficientSize = bcCoefficients.value.size

  override protected val dim: Int = coefficientSize

  require(coefficientSize == numFeaturesPlusIntercept * 2,
    s"Expected $numFeaturesPlusIntercept coefficients but got $coefficientSize")

  @transient private lazy val coefficientsArray = bcCoefficients.value match {
    case DenseVector(values) => values
    case _ => throw new IllegalArgumentException(s"coefficients only supports dense vector but " +
      s"got type ${bcCoefficients.value.getClass}.")
  }

  override def add(instance: Instance): DFMAggregator = instance match {
    case Instance(label, weight, time, features) =>
      require(numFeatures == features.size, s"Dimensions mismatch when adding new instance." +
        s" Expecting $numFeatures but got ${features.size}.")
      require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")
      require(time >= 0.0 && time <= 1.0, s"instance time, $time has to be >= 0.0 and <= 1.0")

      if (weight == 0.0) return this

      val localFeaturesStd = bcFeaturesStd.value
      val localCoefficients = coefficientsArray
      val localGradientArray = gradientSumArray

      val (wcx, wdx) = {
        var sumC = 0.0
        var sumD = 0.0
        features.foreachActive { (index, value) =>
          if (localFeaturesStd(index) != 0.0 && value != 0.0) {
            sumC += localCoefficients(index) * value / localFeaturesStd(index)
            sumD += localCoefficients(index + numFeaturesPlusIntercept) * value / localFeaturesStd(index)
          }
        }
        if (fitIntercept) {
          sumC += localCoefficients(numFeaturesPlusIntercept - 1)
          sumD += localCoefficients(numFeaturesPlusIntercept * 2 - 1)
        }
        (sumC, sumD)
      }

      val p = sigmoid(wcx)
      val l = math.exp(wdx)

      val (multiplierC, multiplierD) = {
        val (multiplierC, multiplierD) = if (label > 0) (p - 1.0, l * time - 1.0) else
          (p * (1.0 - p) * (1.0 - math.exp(-l * time)) / (1.0 - p + p * math.exp(-l * time)),
            (l * p * time * math.exp(-l * time)) / (1.0 - p + p * math.exp(-l * time)))
        (multiplierC * weight, multiplierD * weight)
      }

      features.foreachActive { (index, value) =>
        if (localFeaturesStd(index) != 0.0 && value != 0.0) {
          localGradientArray(index) += multiplierC * value / localFeaturesStd(index)
          localGradientArray(index + numFeaturesPlusIntercept) += multiplierD * value / localFeaturesStd(index)
        }
      }

      if (fitIntercept) {
        localGradientArray(numFeaturesPlusIntercept - 1) += multiplierC
        localGradientArray(numFeaturesPlusIntercept * 2 - 1) += multiplierD
      }

      // TODO: prevent overflow and zero-division
      if (label > 0) {
        lossSum -= weight * (-MLUtils.log1pExp(-wcx) + wdx - math.exp(wdx) * time)
      } else {
        lossSum -= weight * math.log1p(-p + p * math.exp(-l * time))
      }

      weightSum += weight
      this
  }

}
