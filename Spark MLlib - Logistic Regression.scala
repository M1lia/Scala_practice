import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}

import org.apache.spark.sql.SparkSession


object Lab4 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Lab4")
      .master("local[4]")
      .getOrCreate();

    val _df = spark.read
      .option("header","true")
      .option("delimiter",",")

      .option("treatEmptyValuesAsNulls", "true")
      .option("nullValue", "")
      .option("inferschema","true")
      //.schema(schema)
      .csv("C:/Users/IM/IdeaProjects/Proekt/src/main/resources/diabets.csv");
    _df.printSchema()
    _df.show(10)
    val assembler = new VectorAssembler().
      setInputCols(Array(
      "pregnancy","glucose","arterial pressure","thickness of TC","insulin","body mass index","heredity","age")).
      setOutputCol("features")
    val features = assembler.transform(_df)
    features.show(20)
    val _labels =  new StringIndexer().setInputCol("diabet").setOutputCol("label")
    val labels = _labels.fit(features).transform(features)
    labels.show(13)
    val _split = labels.randomSplit(Array(0.6,0.4),seed = 11L)
    val train = _split(0)
    val test = _split (1)
    train.show(10)
    test.show(10)
    val LR = new LogisticRegression()
      .setMaxIter(1000)
      .setRegParam(0.3)
      .setElasticNetParam(0.1)

    val model =LR.fit(train)
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")
    val predictions = model.transform(test)
    predictions.show(200)
    predictions.createOrReplaceTempView("predictions")
    spark.sql("SELECT diabet,prediction FROM predictions Where prediction != \"0.0\"").show(10)
    val countProve = predictions.where("label==prediction").count()
    val count =predictions.count()
    println(s"Count of true predictions: $countProve Total Count: $count")
    val evaluator= new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy = ${accuracy}")



  }

}
