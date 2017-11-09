package dfm.util

import org.apache.spark.internal.Logging

object MLUtils extends Logging {

  /**
    * When `x` is positive and large, computing `math.log(1 + math.exp(x))` will lead to arithmetic
    * overflow. This will happen when `x > 709.78` which is not a very large number.
    * It can be addressed by rewriting the formula into `x + math.log1p(math.exp(-x))` when `x > 0`.
    * @param x a floating-point value as input.
    * @return the result of `math.log(1 + math.exp(x))`.
    */
  def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }
}
