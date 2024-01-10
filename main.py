# This script is used to train a logistic regression model to predict whether a credit application will be approved or not.

import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col, when, trim
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create a SparkSession
spark = SparkSession.builder.appName('credit_approval').getOrCreate()

# Load the data
df = spark.read.csv('credit_approval_dataset.csv', inferSchema=True, header=True)

# Remove the 'loan_id' column
df = df.drop('loan_id')

# strip column names and rows of spaces
for col in df.columns:
    df = df.withColumnRenamed(col, col.replace(' ', ''))
df = df.withColumn('loan_status', trim(df.loan_status))

# Recode 'loan_status' from 'Approved' and 'Rejected' to 1 and 0
df = df.withColumn('loan_status', when(df.loan_status == 'Approved', 1).otherwise(0))

# One-hot encode the categorical features
categorical_columns = ['education', 'self_employed']
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in categorical_columns]
encoders = [OneHotEncoder(inputCol=column+"_index", outputCol=column+"_ohe") for column in categorical_columns]

# Use a pipeline to combine the steps
pipeline = Pipeline(stages=indexers + encoders)
df = pipeline.fit(df).transform(df)

df = df.drop('education', 'self_employed', 'education_index', 'self_employed_index')

# Create a VectorAssembler
feature_columns = df.drop('loan_status').columns
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df = assembler.transform(df)

# Split the data into train and test sets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# Create a VectorAssembler to convert the column to vector format
assembler_standardization = VectorAssembler(inputCols=['income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
                                                       'residential_assets_value', 'commercial_assets_value', 
                                                       'luxury_assets_value', 'bank_asset_value'], 
                                            outputCol='quantitative_features')

# Transform the DataFrames
train_data = assembler_standardization.transform(train_data)
test_data = assembler_standardization.transform(test_data)

# Create a StandardScaler
scaler = StandardScaler(inputCol='quantitative_features', outputCol='quantitative_features_scaled')

# Fit the StandardScaler to the train data and transform both datasets
scalerModel = scaler.fit(train_data)
train_data = scalerModel.transform(train_data)
test_data = scalerModel.transform(test_data)

# Create the model
lr = LogisticRegression(labelCol='loan_status')

# Generate a list of values for regParam
list_of_values = np.logspace(-3, 3, num=7).tolist()
paramGrid = ParamGridBuilder().addGrid(lr.regParam, list_of_values).build()

# cross-validation
crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol="loan_status"),
                          numFolds=5)

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(train_data)

# Make predictions on test data. 
predictions = cvModel.transform(test_data)

# Create an evaluator
evaluator = MulticlassClassificationEvaluator(labelCol='loan_status', metricName='accuracy')

# Compute the classification error on test data
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

# Create a confusion matrix
predictions.groupBy('loan_status', 'prediction').count().show()

# Create evaluators
evaluator1 = BinaryClassificationEvaluator(labelCol="loan_status", metricName='areaUnderROC')
evaluator2 = MulticlassClassificationEvaluator(labelCol='loan_status', metricName='weightedPrecision')
evaluator3 = MulticlassClassificationEvaluator(labelCol='loan_status', metricName='weightedRecall')
evaluator4 = MulticlassClassificationEvaluator(labelCol='loan_status', metricName='accuracy')

# Compute the metrics on test data
auc = evaluator1.evaluate(predictions)
precision = evaluator2.evaluate(predictions)
recall = evaluator3.evaluate(predictions)
accuracy = evaluator4.evaluate(predictions)

print("AUC-ROC = %g" % auc)
print("Precision = %g" % precision)
print("Recall = %g" % recall)
print("Accuracy = %g" % accuracy)