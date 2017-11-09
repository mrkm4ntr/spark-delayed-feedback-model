package dfm.param

import org.apache.spark.ml.param._

trait HasTimeCol extends Params {

  final val timeCol: Param[String] = new Param[String](this, "timeCol", "")

  final def getTimeCol: String = $(timeCol)
}
