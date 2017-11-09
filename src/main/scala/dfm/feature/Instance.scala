package dfm.feature

import org.apache.spark.ml.linalg.Vector

case class Instance(label: Double, weight: Double, time: Double, features: Vector)
