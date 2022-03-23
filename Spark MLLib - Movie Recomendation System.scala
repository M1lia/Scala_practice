import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
object Lab5 {
  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
  case class Movie(movieId: Int, movieName: String, rating: Float)
  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    return Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }
    def main(args: Array[String]): Unit = {
      val spark = SparkSession
        .builder()
        .appName("Lab4")
        .master("local[4]")
        .getOrCreate();
      import spark.implicits._
      val myRating = spark.read.textFile("C:/Users/IM/IdeaProjects/Proekt/src/test/personalRatings.txt")
        .map(parseRating)
        .toDF()
      //C:\Users\IM\IdeaProjects\Proekt\src\test\ratings.dat.txt
      val ratings = spark
        .read.textFile("C:/Users/IM/IdeaProjects/Proekt/src/test/ratings.dat.txt")
        .map(parseRating)
        .toDF()
      val moviesRDD = spark
        .read.textFile("C:/Users/IM/IdeaProjects/Proekt/src/test/movies.dat.txt").map { line =>
        val fields = line.split("::")
        (fields(0).toInt, fields(1))
      }
      ratings.show(10)
      myRating.show(10)
      val numRatings = ratings.distinct().count()
      val numUsers = ratings.select("userId").distinct().count()
      val numMovies = moviesRDD.count()
      val movies = moviesRDD.collect.toMap
      println("Got " + numRatings + " ratings from "
        + numUsers + " users on " + numMovies + " movies.")
      val ratingWithMyRats = ratings.union(myRating)
      val Array(training, test) = ratingWithMyRats.randomSplit(Array(0.5, 0.5))
      val als = new ALS()
        .setMaxIter(3)
        .setRegParam(0.01)
        .setUserCol("userId")
        .setItemCol("movieId")
        .setRatingCol("rating")
      val model = als.fit(training)
      val predictions = model.transform(test)
      val evaluator = new RegressionEvaluator()
        .setMetricName("rmse")
        .setLabelCol("rating")
        .setPredictionCol("prediction")
      val rmse = evaluator.evaluate(predictions)
      println(s"Root-mean-square error = $rmse")
      val myPredictions = model.transform(myRating).na.drop
      val myMovies = myPredictions.map(r => Movie(r.getInt(1), movies(r.getInt(1)), r.getFloat(2))).toDF
      myMovies.show(100)
      ratings.show(10)
      myRating.show(10)
      println("Got " + numRatings + " ratings from "
        + numUsers + " users on " + numMovies + " movies.")
      println(s"Root-mean-square error = $rmse")

    }


}
