# Import necessary PySpark modules
import findspark # Import findspark first
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os # Import os for file path manipulation

# --- Initialize findspark (if not already done via environment variables) ---
# This is crucial if 'pyspark' command is not found.
# It helps Python locate the PySpark installation.
# If you installed PySpark via pip, findspark will usually find it automatically.
# If you downloaded Apache Spark manually, you might need to specify SPARK_HOME
# explicitly, e.g., findspark.init('/path/to/spark-3.x.x-bin-hadoopY.y')
try:
    findspark.init()
    print("findspark initialized successfully.")
except Exception as e:
    print(f"Error initializing findspark: {e}")
    print("Please ensure PySpark is installed or set SPARK_HOME environment variable.")
    exit()


# --- Create a SparkSession ---
# This is the entry point to programming Spark with the Dataset and DataFrame API.
# It allows you to create DataFrames, register DataFrames as tables,
# execute SQL queries, read JSON files, etc.
# The `appName` is a name for your application, which will be shown in the Spark UI.
# `master` specifies the master URL for the cluster; 'local[*]' means run locally
# with as many worker threads as logical cores on your machine.
spark = SparkSession.builder \
    .appName("SimplePySparkWordCount") \
    .master("local[*]") \
    .getOrCreate()

# --- Data Preparation (Creating a dummy text file) ---
# In a real scenario, you would read from HDFS, S3, or a local file system.
# For this example, we'll create a small text file programmatically.
file_name = "sample_text.txt"
try:
    with open(file_name, "w") as f:
        f.write("Hello Spark\n")
        f.write("Spark is powerful\n")
        f.write("Hello world\n")
        f.write("This is a simple example for Spark\n")
    print(f"Created '{file_name}' for demonstration in current directory.")
except IOError as e:
    print(f"Error creating {file_name}: {e}")
    spark.stop() # Stop Spark session if file creation fails
    exit()

# --- Read the text file into an RDD (Resilient Distributed Dataset) ---
# An RDD is a fundamental data structure of Spark. It is a distributed collection
# of elements that can be operated on in parallel.
# `textFile` reads each line of the file as an element in the RDD.
# Ensure the file path is correct for Spark to read.
current_dir = os.getcwd()
file_path = os.path.join(current_dir, file_name)

lines_rdd = spark.sparkContext.textFile(file_path)
print("\n--- Original Lines (RDD) ---")
# `collect()` brings all RDD elements to the driver program.
# Be cautious with large datasets as this can cause OutOfMemory errors.
for line in lines_rdd.collect():
    print(line)

# --- Transformation: Split lines into words and flatten ---
# `flatMap` is a transformation that flattens the RDD after applying a function.
# Here, it splits each line by space and creates a new RDD with individual words.
words_rdd = lines_rdd.flatMap(lambda line: line.lower().split(" "))
print("\n--- Individual Words (RDD) ---")
for word in words_rdd.collect():
    print(word)

# --- Transformation: Pair each word with a count of 1 ---
# `map` is a transformation that applies a function to each element of the RDD.
# Here, it creates key-value pairs (word, 1).
word_pairs_rdd = words_rdd.map(lambda word: (word, 1))
print("\n--- Word Pairs (RDD) ---")
for pair in word_pairs_rdd.collect():
    print(pair)

# --- Action: Count the occurrences of each word ---
# `reduceByKey` is a transformation that aggregates values for each key.
# Here, it sums the counts (1s) for each unique word.
# `collectAsMap()` is an action that returns the key-value pairs as a Python dictionary.
word_counts = word_pairs_rdd.reduceByKey(lambda a, b: a + b).collectAsMap()
print("\n--- Word Counts (Result) ---")
for word, count in sorted(word_counts.items()): # Sort for consistent output
    print(f"'{word}': {count}")

# --- Using DataFrames for Word Count (more modern Spark approach) ---
# Spark DataFrames are a more optimized and common way to work with structured data.
# They provide a richer set of operations than RDDs and are more performant.
print("\n--- Word Count using DataFrames ---")
df = spark.read.text(file_path) \
    .select(F.explode(F.split(F.lower(F.col("value")), " ")).alias("word")) \
    .groupBy("word") \
    .count() \
    .orderBy("word") # Order by word for consistent output

df.show() # Display the DataFrame content

# --- Stop the SparkSession ---
# It's good practice to stop the SparkSession when your application is done.
spark.stop()
print("\nSparkSession stopped.")

