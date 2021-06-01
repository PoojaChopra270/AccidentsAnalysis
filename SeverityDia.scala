package com.sundogsoftware.spark

import org.apache.log4j._
import org.apache.spark._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql._

import scala.math.max



/** Find the maximum severity for for an accident ID */
object SeverityDia {

  case class Characteristics(Num_Acc: Long, Day: Int, Month: Int, Year: Int, Light: Int, agglomeration: Int, intersection: Int, AtmosphericCondition: Int, CollisionType: Int, Hr: Int)

  case class Places(Num_Acc: Long, RoadCategory: Int, TrafficRegime: Int, NoOfLane: Int, CycleLane: Int, LongitudinalProfile: Int, NoOfUpstream: Int, DistanceFromUpstream: Int, PlanLayout: Int, SurfaceCondition: Int, Infrastructure: Int, SituationAccident: Int, NearSchool: Int)

  case class Severity(Num_Acc: Long, inj: Int)

  case class Safety(Num_Acc: Long, vehicle_id: String, victim: Int, injury:Int, Gender:Int, YearBorn:Int, Safety_equipment_1:Int, Safety_equipment_2:Int, Safety_equipment_3:Int, Pedestrian_location:Int, Pedestrian_Action: Int, Alone_Pedestrian:Int)

  case class Vehicles(Num_Acc_d: Long, vehicle_id: String, vehicle_type: Int, FixedObstacle: Int, MovableObstacle: Int, Shock: Int, Maneuver: Int, Motor: Int )

  def parseLine(line: String): (Long, Int, Int, Int, Int,Int) = {
    val fields = line.split(",")
    val Num_Acc = fields(0).toLong
    val severity = fields(5).replace("\"", "").replace("2", "tmp").replace("4", "2").replace("tmp", "4")toInt
    val equipment_1 = fields(9).replace("\"", "").replace("-1", "0")toInt
    val equipment_2 = fields(10).replace("\"", "").replace("-1", "0")toInt
    val equipment_3 = fields(11).replace("\"", "").replace("-1", "0")toInt
    val user_type = fields(4).replace("\"", "").replace("-1", "0")toInt

    (Num_Acc, severity, equipment_1, equipment_2, equipment_3,user_type)
  }


  def mapper(line: String): Severity = {
    val f = line.split(',')
    val severity: Severity = Severity(f(0).replace("\"", "")toLong,
      f(5).replace("\"", "").replace("-1", "0")toInt)
    severity
  }

  def mapper1(lines1: String): Characteristics = {
    val fields = lines1.split(',')
    val characteristic: Characteristics = Characteristics(fields(0).replace("\"", "") toLong,
      fields(1).replace("\"", "").replace("-1", "0") toInt,
      fields(2).replace("\"", "").replace("-1", "0") toInt,
      fields(3).replace("\"", "").replace("-1", "0") toInt,
      fields(5).replace("\"", "").replace("-1", "0") toInt,
      fields(8).replace("\"", "").replace("-1", "0") toInt,
      fields(9).replace("\"", "").replace("-1", "0") toInt,
      fields(10).replace("\"", "").replace("-1", "0") toInt,
      fields(11).replace("\"", "").replace("-1", "0") toInt,
      fields(4).replace("\"", "").take(2).replace(".","").replace("-1", "0") toInt)

    characteristic
  }

  def mapper2(lines2: String): Places = {
    val field = lines2.split(',')
    val places: Places = Places(field(0).replace("\"", "") toLong,
      field(1).replace("\"", "").replace("-1", "0") toInt,
      field(5).replace("\"", "").replace("-1","0")toInt,
      field(6).replace("\"", "").replace("(", "").replace(")", "").replace("-1", "0") toInt,
      field(7).replace("\"", "").replace("(", "").replace(")", "").replace("-1", "0")toInt,
      field(8).replace("\"", "").replace("(", "").replace(")", "").replace("-1", "0") toInt,
      field(9).replace("\"", "").replace("(", "").replace(")", "").replace("-1", "0") toInt,
      field(10).replace("\"", "").replace("(", "").replace(")", "").replace("-1", "0") toInt,
      field(11).replace("\"", "").replace("(", "").replace(")", "").replace("-1", "0") toInt,
      field(14).replace("\"", "").replace("(", "").replace(")", "").replace("-1", "0") toInt,
      field(15).replace("\"", "").replace("(", "").replace(")", "").replace("-1", "0") toInt,
      field(16).replace("\"", "").replace("(", "").replace(")", "").replace("-1", "0") toInt,
      field(17).replace("\"", "").replace("(", "").replace(")", "").replace("-1", "0") toInt)

    places
  }


  def mapper3(lines: String): Safety = {
    val fieldSafe =lines.split(',')
    val safety: Safety =Safety(fieldSafe(0).replace("\"", "")toLong,
      fieldSafe(1).replace("\"", "").replace("-1", "0"),
      fieldSafe(4).replace("\"", "").replace("-1", "0")toInt,
      fieldSafe(5).replace("\"", "").replace("-1", "0").replace("4", "tmp").replace("2", "4").replace("tmp","2").toInt,
      fieldSafe(6).replace("\"", "").replace("-1", "0")toInt,
      fieldSafe(7).replace("\"", "").replace("-1", "0")toInt,
      fieldSafe(9).replace("\"", "").replace("-1", "0")toInt,
      fieldSafe(10).replace("\"", "").replace("-1", "0")toInt,
      fieldSafe(11).replace("\"", "").replace("-1", "0")toInt,
      fieldSafe(12).replace("\"", "").replace("-1", "0")toInt,
      fieldSafe(13).replace("\"", "").replace("B","0").replace("A","10").replace("-1", "0")toInt,
      fieldSafe(14).replace("\"", "").replace("-1", "0")toInt)
    safety
  }


  def mapper4(lines3: String) :Vehicles ={
    val vFields =lines3.split(',')
    val vehicles:Vehicles =Vehicles(vFields(0).replace("\"", "")toLong,
      vFields(1).replace("\"", "").replace("-1", "0"),
      vFields(4).replace("\"", "").replace("-1", "0")toInt,
      vFields(5).replace("\"", "").replace("-1", "0")toInt,
      vFields(6).replace("\"", "").replace("-1", "0")toInt,
      vFields(7).replace("\"", "").replace("-1", "0")toInt,
      vFields(8).replace("\"", "").replace("-1", "0")toInt,
      vFields(9).replace("\"", "").replace("-1", "0")toInt)

    vehicles
  }

  /** Our main function where the action happens */
  def main(args: Array[String]) {

    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Create a SparkContext using every core of the local machine
    val sc = new SparkContext("local[*]", "MaxSeverity")

    val spark = SparkSession
      .builder
      .appName("SparkSQL")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "file:///C:/temp") // Necessary to work around a Windows bug in Spark 2.0.0; omit if you're not on Windows.
      .getOrCreate()

    import spark.implicits._
    val lines = sc.textFile("/home/hduser/data/users_noheader.csv")
    val injured = lines.map(parseLine)

    val lines1 = spark.sparkContext.textFile("/home/hduser/data/characteristics_noheader.csv")
    val characteristic = lines1.map(mapper1).toDS().cache()

    val lines2 = spark.sparkContext.textFile("/home/hduser/data/places_noheader.csv")
    val places = lines2.map(mapper2).toDS().cache()

    val lines3 = spark.sparkContext.textFile("/home/hduser/data/vehicles_noheader.csv")
    val vehicles = lines3.map(mapper4).toDS().cache()

    ///Counting number of injured person per accident
    val acc_ct = injured.map(x => (x._1,1))
    val Acc_ct = acc_ct.reduceByKey((x,y) => x+y)
    val Acc_count_df= Acc_ct.toDS().cache()
    val newName = Seq("Num_Acc", "ct")
    val A_C = Acc_count_df.toDF(newName: _*)

    /// checking maximum injury in an accident.
    // For example one person slightly injured and another died then max severity is killed
    val MaxSeverity = injured.map(x => (x._1, x._2.toInt))
    val Severity_df = MaxSeverity.reduceByKey((x, y) => max(x, y))
    val results = Severity_df.toDS().cache()
    val newNames = Seq("Num_Acc", "Max_Severity")
    val Sev = results.toDF(newNames: _*)

    /// Testing the formed table containing max severity and no of injured person per accident
    // Creating a view from a file
    val injury = lines.map(mapper).toDS().cache()
    injury.createOrReplaceTempView("injury")

    val injury_safety = lines.map(mapper3).toDS().cache()
    injury_safety.createOrReplaceTempView("safety")

    val df= characteristic.join(Sev,usingColumn = "Num_Acc").join(A_C,usingColumn = "Num_Acc").join(places,usingColumn = "Num_Acc").join(injury_safety, usingColumn = "Num_Acc").join(vehicles,usingColumn = "vehicle_id")
    println("Here is our inferred schema of main table:")
    // forming a view of dataframe
    df.printSchema()
    df.createOrReplaceTempView("details")


    val count_per_accident = spark.sql("Select distinct(Num_Acc),ct FROM details where Num_Acc in(201900000001,201900000002,201900000003,201900000004) order by Num_Acc")
    val Count_per_accident = count_per_accident.collect()
    println(" ")
    println("Testing the formed table containing max severity and no of injured person per accident ")
    println(" Count obtained from rdd for 4 accident:")
    Count_per_accident.foreach(println)

    val test_count = spark.sql("Select distinct(Num_Acc),count(*) FROM injury group by Num_Acc having Num_Acc in(201900000001,201900000002,201900000003,201900000004) order by Num_Acc")
    val Test_count = test_count.collect()
    println(" Count obtained by grouping Num_Acc in a table created from the original data file-main_users:")
    Test_count.foreach(println)


    val test_max=spark.sql("SELECT distinct(Num_Acc), Max_Severity FROM details where Num_Acc in(201900000166,201900000181,201900000414,201900000463) order by Num_Acc")
    val Test_max = test_max.collect()
    println("")
    println("")
    println("Testing max severity in an accident")
    println(" Max function used on rdd:")
    Test_max.foreach(println)


    val inj = spark.sql("SELECT Num_Acc,max(inj) from injury group by Num_Acc having Num_Acc in (201900000166,201900000181,201900000414,201900000463) order by Num_Acc")
    val Inj = inj.collect()
    println("Max function used on a table created from the original data file-main_users:")
    Inj.foreach(println)



    println("")
    println("")
    println("Checking relationship between safety equipment and severity using RDD")
    println("With safety equipment ")

    val equip = injured.filter(x => x._3 > 0 || x._4 > 0 || x._5 >0)
    val total_equip = equip.count().toLong
    val t_safety = equip.map(x => (x._2, 1))
    val safety_count = t_safety.reduceByKey( (x,y) => x+y)
    val results_c = safety_count.collect()
    println("Severity:Percentage of people with at least one safety equipment")
    for (results_c <- results_c.sorted) {
      val severity = results_c._1
      val count = (results_c._2.toFloat/total_equip)*100
      println(s"$severity : $count")
    }

    val no_equip = injured.filter(x => (x._3 == 0 && x._4 == 0 && x._5 == 0))
    val no_total_equip = no_equip.count().toLong
    val t_no_safety = no_equip.map(x => (x._2, 1))
    val no_safety_count = t_no_safety.reduceByKey( (x,y) => x+y)
    val results_nc = no_safety_count.collect()
    println("Without safety equipment")
    println("Severity:Percentage")
    for (results_nc <- results_nc.sorted) {
      val nseverity = results_nc._1
      val ncount = (results_nc._2.toFloat/no_total_equip)*100
      println(s"$nseverity : $ncount")
    }


    ///Checking relationship severity and safety equipment
    val people_with_safety = spark.sql("select count(*) from safety where Safety_equipment_1>0 or Safety_equipment_1>0 or Safety_equipment_1>0")
    val testing_val = people_with_safety.collect()(0).getLong(0)
    val equipment = spark.sql(s"select (count(*)/$testing_val)*100,injury from safety  where Safety_equipment_1>0 or Safety_equipment_1>0 or Safety_equipment_1>0 group by injury order by injury")
    val Equipment = equipment.collect()
    println("")
    println("")
    println("Testing with safety equipment using sql")
    println("Percentage,severity")
    Equipment.foreach(println)

    val people_withno_safety = spark.sql("select count(*) from safety where Safety_equipment_1=0 and Safety_equipment_1=0 and Safety_equipment_1=0 ")
    val testingno_val = people_withno_safety.collect()(0).getLong(0)
    val no_equipment = spark.sql( sqlText = s"select (count(*)/$testingno_val)*100,injury from safety  where Safety_equipment_1=0 and Safety_equipment_1=0 and Safety_equipment_1=0 group by injury order by injury")
    val No_equipment = no_equipment.collect()
    println("Testing with safety without equipment using sql")
    println("Percentage,Severity")
    No_equipment.foreach(println)

    /// checking months having top most accidents
    val month_val=spark.sql("SELECT Month,count(*) as ct  FROM details GROUP BY Month ORDER BY ct desc")
    val Month_val = month_val.collect()
    println("")
    println("Now data is ready.Checking time parameter of the accident data")
    println("checking months having top most accidents")
    println("Month,count")
    Month_val.foreach(println)

    //checking hours having top most accidents
    val hour_val=spark.sql("SELECT Hr,count(*) as ct  FROM details GROUP BY Hr ORDER BY ct desc")
    val Hour_val = hour_val.collect()
    println(" ")
    println("checking hours having top most accidents")
    println("Hour,count")
    Hour_val.foreach(println)


    val light_val=spark.sql("SELECT light,Max_Severity, count(*) as ct  FROM details GROUP BY Light, Max_Severity ORDER BY Max_Severity,ct desc")
    val Light_val = light_val.collect()
    println(" ")
    println("checking hours having top most accidents")
    println("light,severity,count")
    Light_val.foreach(println)

    val atm_val=spark.sql("SELECT AtmosphericCondition,Max_Severity, count(*) as ct  FROM details GROUP BY AtmosphericCondition, Max_Severity ORDER BY Max_Severity,ct desc")
    val ATM_val = atm_val.collect()
    println(" ")
    println("checking hours having top most accidents")
    println("AtmosphericCondition,severity,count")
    ATM_val.foreach(println)

    val surface_val=spark.sql("SELECT SurfaceCondition,Max_Severity, count(*) as ct  FROM details GROUP BY SurfaceCondition, Max_Severity ORDER BY Max_Severity,ct desc")
    val Surface_val = surface_val.collect()
    println(" ")
    println("checking surface and accident frequency:")
    println("Surface,severity,count")
    Surface_val.foreach(println)

    val gender_profile=spark.sql("SELECT gender, injury, count(*)  as ct FROM details where victim=1 GROUP BY gender, injury ORDER BY  injury,ct desc")
    val Gender_profile = gender_profile.collect()
    println("")
    println("checking gender and count of accidents")
    println("Gender, Injury, Count")
    Gender_profile.foreach(println)


    val age_profile=spark.sql("SELECT (Year-YearBorn) as age, count(*)  as ct FROM details where victim=1 GROUP BY age ORDER BY ct desc")
    val Age_profile = age_profile.collect()
    println("")
    println("checking Age and count of accidents")
    println("Age,Count")
    Age_profile.foreach(println)


    val maneuverType=spark.sql("SELECT Maneuver,count(*) as ct  FROM details GROUP BY Maneuver ORDER BY ct desc limit 10")
    val ManeuverType = maneuverType.collect()
    println("")
    println("Checking type of Maneuver and injury severity")
    println("Maneuver,Max_Severity,count")
    ManeuverType.foreach(println)

    println("")


    println("Measuring time for machine learning...")
    val startTimeMillis = System.currentTimeMillis()

    println("Up-sampling...")

    val killed = df.filter("injury=4")
    val hospitalized = df.filter("injury=3")
    val slightlyInjured = df.filter("injury=2")
    val unharmed = df.filter("injury=1")

    val upSampledKilled = killed.sample( true,fraction=12)
    val sampled_df=hospitalized.unionAll(upSampledKilled).unionAll(slightlyInjured).unionAll(unharmed)

    val df2 = sampled_df.drop("vehicle_id","Num_Acc_d","Num_Acc")

    println("One-hotencoding...")
    val encoder = new OneHotEncoder()
      .setInputCols(Array("Month","Light","agglomeration","intersection","AtmosphericCondition","CollisionType","Hr","RoadCategory",
        "TrafficRegime","NoOfLane","CycleLane","LongitudinalProfile","PlanLayout","victim","Max_Severity","Pedestrian_location","Pedestrian_Action","Alone_Pedestrian",
        "SurfaceCondition","Infrastructure","SituationAccident","NearSchool","Gender","YearBorn","Safety_equipment_1","Safety_equipment_2",
        "Safety_equipment_3", "vehicle_type","FixedObstacle","MovableObstacle","Shock","Maneuver","Motor"))
      .setOutputCols(Array("Month_vec","Light_vec","agglomeration_vec","intersection_vec","AtmosphericCondition_vec","CollisionType_vec", "Hr_vec","RoadCategory_vec",
        "TrafficRegime_vec", "NoOfLane_vec", "CycleLane_vec","LongitudinalProfile_vec","PlanLayout_vec","victim_vec","Max_Severity_vec","Pedestrian_location_vec","Pedestrian_Action_vec","Alone_Pedestrian_vec",
        "SurfaceCondition_vec", "Infrastructure_vec",  "SituationAccident_vec", "NearSchool_vec", "Gender_vec", "YearBorn_vec", "Safety_equipment_1_vec", "Safety_equipment_2_vec",
        "Safety_equipment_3_vec", "vehicle_type_vec", "FixedObstacle_vec", "MovableObstacle_vec", "Shock_vec", "Maneuver_vec",   "Motor_vec"))
    val model = encoder.fit(df2)
    val encoded = model.transform(df2)


    val col_vectors= Array("Month_vec","Light_vec","agglomeration_vec","intersection_vec","AtmosphericCondition_vec","CollisionType_vec",
      "Hr_vec","RoadCategory_vec","TrafficRegime_vec", "LongitudinalProfile_vec", "Pedestrian_location_vec","Pedestrian_Action_vec","Alone_Pedestrian_vec",
      "PlanLayout_vec", "SurfaceCondition_vec", "Infrastructure_vec","ct","Max_Severity_vec",
      "SituationAccident_vec", "Gender_vec", "YearBorn_vec", "Safety_equipment_1_vec", "Safety_equipment_2_vec","victim_vec",
      "Safety_equipment_3_vec", "vehicle_type_vec", "FixedObstacle_vec", "MovableObstacle_vec", "Shock_vec", "Maneuver_vec",   "Motor_vec")

    println("predictor and target separation....")

    val assembler2 = new VectorAssembler()
      .setInputCols(col_vectors)
      .setOutputCol("features")
    val featureDf = assembler2.transform(encoded)
    val indexer = new StringIndexer()
      .setInputCol("injury")
      .setOutputCol("label")

    val labelDf = indexer.fit(featureDf).transform(featureDf)

    println("test and train splitting..")
    val seed = 0
    val Array(trainingData, testData) = labelDf.randomSplit(Array(0.7, 0.3), seed)
    trainingData.cache()
    // train Random Forest model with training data set

    val randomForestClassifier = new RandomForestClassifier().setNumTrees(8).setSeed(seed)


    println("Building model...")
    val randomForestModel = randomForestClassifier.fit(trainingData)
    println("predicting...")
    val predictionRF = randomForestModel.transform(testData)

    println("evaluating...")
    val evaluator1 = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("weightedRecall")


    val recall = evaluator1.evaluate(predictionRF)
    println("")
    println("weightedRecall: "+  recall)

    val evaluator3 = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("weightedPrecision")

    val weightedPrecision = evaluator3.evaluate(predictionRF)
    println("weightedPrecision: "+  weightedPrecision)

    val endTimeMillis = System.currentTimeMillis()

    val durationMins = (endTimeMillis - startTimeMillis) / 1000.toFloat
    println(" ")
    println("Model Building and pre-processing took:"+ durationMins + " secs")

    spark.stop()
  }
}

