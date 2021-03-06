import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans
object Lab6 {
  case class Uber(dt: String, lat: Double, lon: Double, base: String) extends Serializable

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Lab4")
      .master("local[4]")
      .getOrCreate();
    import spark.implicits._
    val schema = StructType(Array(
      StructField("dt", TimestampType, true),
      StructField("lat", DoubleType, true),
      StructField("lon", DoubleType, true),
      StructField("base", StringType, true)
    ))
    val uberData = spark.read
      .option("header", "true")
      .option("inferSchema", "false")
      .schema(schema)
      .csv("C:/Users/IM/IdeaProjects/Proekt/src/test/uber.csv")
      .as[Uber]
    uberData.show()
    uberData.printSchema()
    uberData.createOrReplaceTempView("uber")
    spark.sql("SELECT base, COUNT(base) as cnt FROM uber GROUP BY base").show()
    spark.sql("SELECT date(dt), COUNT(base) as cnt FROM uber GROUP BY date(dt), base ORDER BY 1").show()
    val featureCols = Array("lat", "lon")
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
    val uberFeatures = assembler.transform(uberData)
    val Array(trainingData, testData) = uberFeatures.randomSplit(Array(0.7, 0.3), 5043)
    val kmeans = new KMeans()
      .setK(20)
      .setFeaturesCol("features")
      .setMaxIter(5)
    val model = kmeans.fit(trainingData)
    println("Final Centers: ")
    model.clusterCenters.foreach(println)
    val predictions = model.transform(testData)
    predictions.show
    predictions.createOrReplaceTempView("uber")
    predictions.select(month($"dt").alias("month"), dayofmonth($"dt").alias("day"), hour($"dt").alias("hour"), $"prediction")
      .groupBy("month", "day", "hour", "prediction")
      .agg(count("prediction")
        .alias("count"))
      .orderBy("day", "hour", "prediction").show
    predictions.select(hour($"dt").alias("hour"), $"prediction")
      .groupBy("hour", "prediction").agg(count("prediction")
      .alias("count"))
      .orderBy(desc("count"))
      .show
    predictions.groupBy("prediction").count().show()

    spark.sql("select prediction, count(prediction) as count from uber group by prediction").show

    spark.sql("select hour(uber.dt) as hr,count(prediction) as ct FROM uber group By hour(uber.dt)").show
    val res = spark.sql("select dt, lat, lon, base, prediction as cid FROM uber where prediction = 1")
    res.show()
    res.coalesce(1).write.format("json").save("./data/uber.json")






  }

}
