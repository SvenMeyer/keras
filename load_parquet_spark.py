# http://spark.apache.org/docs/latest/sql-programming-guide.html

from pyspark import SparkContext
from pyspark.sql import SparkSession
import os

# os.environ["PYSPARK_SUBMIT_ARGS"] = ("--master spark://localhost:7077")
#  "--packages com.databricks:spark-csv_2.11:1.3.0 pyspark-shell"


sc = SparkContext()

home = os.path.expanduser("~")
dir  = "/ML_DATA/"
filename = "part-r-00000-67ebd6f0-bfb4-42e0-b516-d7aaa77cbcb8.snappy.parquet"
datafile = home + dir + filename

print("open file : ", datafile)

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


spark_df = spark.read.parquet(datafile)

print("type(spark_df) = ", type(spark_df) )
print(spark_df.count())
print(spark_df.columns)
