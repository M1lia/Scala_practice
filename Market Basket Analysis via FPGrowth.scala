import org.apache.spark.sql.SparkSession
import com.crealytics.spark.excel._
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.sql.types._
object Lab3 {
  def main(args: Array[String]): Unit = {
    val dataSetSchema = StructType(Array(
      StructField("InvoiceNo", StringType, true),
      StructField("StockCode", StringType, true),
      StructField("Description", StringType, true),
      StructField("Quantity", IntegerType, true),
      StructField("InvoiceDate", StringType, true),
      StructField("UnitPrice", DoubleType, true),
      StructField("CustomerID", IntegerType, true),
      StructField("Country", StringType, true)))
    val spark = SparkSession
      .builder()
      .appName("Lab3")
      .master("local[4]")
      .getOrCreate();
    //val df = spark.read.excel(header = true).load("C:/Users/IM/IdeaProjects/Proekt/src/main/resources/Online Retail.xlsx")
    //df.printSchema()
    val df2 = spark.read
      .option("header", "true")
      .option("delimiter", ";")
      .option("treatEmptyValuesAsNulls", "true")
      .option("nullValue", "")
      .schema(dataSetSchema)
      .csv("C:/Users/IM/IdeaProjects/Proekt/src/main/resources/Onlineret.csv")
    df2.printSchema()
    df2.createOrReplaceTempView("fulldata")
    df2.show(10)
    val alltrans = spark.sql("SELECT StockCode,CustomerID FROM fulldata Where CustomerID IS NOT NULL AND StockCode IS NOT NULL")
    alltrans.show(11)
    val alltransv2 = alltrans.rdd.map(s => (s(1).toString, s(0).toString)).groupByKey()
    alltransv2.take(3).foreach(print)
    val alltransv3 = alltransv2.map(s => s._2.toArray.distinct)
    val _dictionary = spark.sql("SELECT StockCode,Description FROM fulldata Where StockCode IS NOT NULL AND Description IS NOT NULL ")
    _dictionary.show(12)
    val dictionary = _dictionary.rdd.map(s=>(s(0),s(1))).collect().toMap
    dictionary.take(10).foreach(print)
    alltransv3.map(s=>s.mkString(",")).take(2).foreach(print)
    //print(alltransv3.map(line=>(dictionary(line(0)),dictionary(line(1)))).first())
    val fgp = new FPGrowth()
      .setMinSupport(0.03)
    val model = fgp.run(alltransv3)
    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
    }
    val minConfidence = 0.3
    val _rules = model.generateAssociationRules(minConfidence)
        _rules.collect().foreach { rule =>
      println(
        rule.antecedent.map(s=>dictionary(s)).mkString("[", ",", "]")
          + " => " + rule.consequent.map(s=>dictionary(s)).mkString("[", ",", "]")
          + ", " + rule.confidence)
    }
  }
}

