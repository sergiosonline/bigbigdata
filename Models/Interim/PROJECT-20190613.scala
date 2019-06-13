//Copy file from local directory to HDFS
hdfs dfs -copyFromLocal /home/betancourt_serest/train.csv .

//file in HDFS
val file_location1="/user/betancourt_serest/train.csv"
//val df = spark.read.format("csv").option("header","true").load(file_location1)
val df1 = spark.read.format("csv").option("inferSchema", "true").option("header","true").load(file_location1)

//HDFS path: /user/betancourt_serest/train.csv


///// Checking distinct values for every column

// approximate counts
import org.apache.spark.sql.functions._
val exprs1 = df1.columns.map((_ -> "approx_count_distinct")).toMap
df1.agg(exprs1).show()

// exact counts and better output
//val exprs2 = df1.columns.map(x => countDistinct(x).as(x))
//df.agg(exprs2.head, exprs2.tail: _*).show



///// Checking number of null values for every column

//df1.select(df1.columns.map(c => count(predicate(col(c))).as(s"nulls column $c")): _*).show()
//private def predicate(c: Column) = {c.isNull || c === "" || c.isNaN || c === "-" || c === "NA"}


val col=df1.columns
var df1Array=col.map(colmn=>df1.select(lit(colmn).as("colName"),sum(when(df1(colmn).isNull || df1(colmn)==="" || df1(colmn)==="-" || df1(colmn).isNaN,1).otherwise(0)).as("missingValues")))

df1Array.tail.foldLeft(df1Array.head)((acc,itr)=>acc.union(itr)).show(false)



///// Rebalancing the data (1/10)
val DFnodetections = df1.filter($"HasDetections"===0)
val DFdetections = df1.filter($"HasDetections"=!=0)
val DFsmalldetections = DFdetections.orderBy(rand()).limit(447000)
//DFsmalldetections.agg(count("HasDetections")).show()
val DFimbalanced = DFnodetections.union(DFsmalldetections)


///// Copy files from server to Google Cloud Storage
gsutil cp ./data_imbalanced.csv gs://mie1628-bigbigdata/data/data_imbalanced.csv


\\Loading randomized data from HDFS
val file_location2="/user/betancourt_serest/data_imbalanced.csv"
\\val df = spark.read.format("csv").option("header","true").load(file_location1)
val df2 = spark.read.format("csv").option("inferSchema", "true").option("header","true").load(file_location2)
val DFimbalrandom = DFimbalrenamed.orderBy(rand())

//////Saving to default HDFS folder
DFrandomized.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("data_randomized.csv")


/////Loading randomized, imbalanced data
hdfs dfs -copyFromLocal /home/betancourt_serest/data/data_randomized.csv .
val file_location3="/user/betancourt_serest/data_randomized.csv"
val df3 = spark.read.format("csv").option("inferSchema", "true").option("header","true").load(file_location3)

//////4'909,591 records initially. 80-20 split


val col=df3.columns
var df3Array=col.map(colmn=>df3.select(lit(colmn).as("colName"),sum(when(df3(colmn).isNull || df3(colmn)==="" || df3(colmn)==="-" || df3(colmn).isNaN,1).otherwise(0)).as("missingValues")))

df3Array.tail.foldLeft(df3Array.head)((acc,itr)=>acc.union(itr)).show(false)

//counting missing values by one column
//df3.filter(df3("Census_ProcessorCoreCount").isNull || df3("Census_ProcessorCoreCount") === "" || df3("Census_ProcessorCoreCount").isNaN).count()



/////// Model pipelines


/////// 1. Naive logistic regression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

val tokenizer = new Tokenizer()
  .setInputCol("text")
  .setOutputCol("words")

val hashingTF = new HashingTF()
  .setNumFeatures(1000)
  .setInputCol(tokenizer.getOutputCol)
  .setOutputCol("features")


val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.001)
  
val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

// Fit the pipeline to training dataframe
val model = pipeline.fit(training)

// Make predictions on test dataframe

model.transform(test)
  .select("id", "text", "probability", "prediction")
  .collect()
  .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
    println(s"($id, $text) --> prob=$prob, prediction=$prediction")
  }

val d = model.transform(test)
d.show()



/////// 2. Random Forests
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// COMMAND ----------

// Load the data stored as a DataFrame
//google data/mllib/sample_libsvm_data.txt

val data = spark.read.format("libsvm").load("/FileStore/tables/forest.txt")
data.count()
data.printSchema()
data.show()

// COMMAND ----------

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)

// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a RandomForest model.
val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)

// COMMAND ----------

// Details about the model and its settings
println(s"Model 1 was fit using parameters: ${rf.extractParamMap}")

// Print out the parameters, documentation, and any default values.
println(s"Random Forest parameters:\n ${rf.explainParams()}\n")
//println(rf.getParam())



/////// 3. Boosted Trees (adaboost, xgBoost)
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}