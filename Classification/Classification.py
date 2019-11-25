#!/usr/bin/env python
# coding: utf-8

# # Binary classification using Logistic Regression on Diabetes dataset

# In[2]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()


# In[19]:


# reading csv file
raw_data = spark.read.format("csv").option("header","true").option("inferSchema", "true").load("diabetes2.csv")


# In[20]:


raw_data.columns


# In[21]:


raw_data.show(5)


# In[22]:


raw_data.describe().select("Summary","SkinThickness","Insulin").show()


# In[23]:


raw_data.describe().select("Summary","BMI","DiabetesPedigreeFunction","Age").show()


# In[24]:


raw_data.describe().select("Summary","Pregnancies","Glucose","BloodPressure").show()


# In[25]:


import pyspark.sql.functions as F


# In[26]:


import numpy as np


# In[27]:


# updating values 0 with np.nan
raw_data=raw_data.withColumn("Glucose",F.when(raw_data.Glucose==0,np.nan).otherwise(raw_data.Glucose))
raw_data=raw_data.withColumn("BloodPressure",F.when(raw_data.BloodPressure==0,np.nan).otherwise(raw_data.BloodPressure))
raw_data=raw_data.withColumn("SkinThickness",F.when(raw_data.SkinThickness==0,np.nan).otherwise(raw_data.SkinThickness))
raw_data=raw_data.withColumn("BMI",F.when(raw_data.BMI==0,np.nan).otherwise(raw_data.BMI))
raw_data=raw_data.withColumn("Insulin",F.when(raw_data.Insulin==0,np.nan).otherwise(raw_data.Insulin))
raw_data.select("Insulin","Glucose","BloodPressure","SkinThickness","BMI").show(5)


# In[ ]:





# In[28]:


# Filling nan with mean
from pyspark.ml.feature import Imputer
imputer=Imputer(inputCols=["Glucose","BloodPressure","SkinThickness","BMI","Insulin"],
                outputCols=["Glucose","BloodPressure","SkinThickness","BMI","Insulin"])
model=imputer.fit(raw_data)
raw_data=model.transform(raw_data)
raw_data.show(5)


# In[29]:


# applying vector assembler
cols=raw_data.columns
cols.remove("Outcome")
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['Pregnancies','Glucose','BloodPressure',
                                       'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'],
                            outputCol="features")
raw_data=assembler.transform(raw_data)
raw_data.select("features").show()


# In[30]:


# applying standard scaler
from pyspark.ml.feature import StandardScaler
standardscaler=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
raw_data=standardscaler.fit(raw_data).transform(raw_data)
raw_data.select("features","Scaled_features").show(5)


# In[32]:


raw_data.select("Scaled_features").show(5,truncate=False)


# In[33]:


# Splitting the dataset
train, test = raw_data.randomSplit([0.75, 0.25], seed=0.8)


# In[34]:


# checking imbalance in training set
dataset_size=float(train.select("Outcome").count())
numPositives=train.select("Outcome").where('Outcome == 1').count()
per_ones=(float(numPositives)/float(dataset_size))*100
numNegatives=float(dataset_size-numPositives)
print('The number of ones are {}'.format(numPositives))
print('Percentage of ones are {}'.format(per_ones))


# In[35]:


# calculating balance ratio
BalancingRatio= numNegatives/dataset_size
print('BalancingRatio = {}'.format(BalancingRatio))


# In[37]:


# adding classWeights column in train set
train=train.withColumn("classWeights", F.when(train.Outcome == 1,BalancingRatio).otherwise(1-BalancingRatio))
train.select("classWeights").show(5)


# In[38]:


# Feature selection using chisquareSelector
from pyspark.ml.feature import ChiSqSelector
css = ChiSqSelector(featuresCol='Scaled_features',outputCol='Aspect',labelCol='Outcome',fpr=0.05)
train=css.fit(train).transform(train)
test=css.fit(test).transform(test)
test.select("Aspect").show(5,truncate=False)


# In[39]:


# fitting the logistic regression model
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="Outcome", featuresCol="Aspect",weightCol="classWeights",maxIter=10)
model=lr.fit(train)
predict_train=model.transform(train)
predict_test=model.transform(test)
predict_test.select("Outcome","prediction").show()


# In[41]:


predict_test.show(1)


# In[43]:


# evaluating
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="Outcome")
predict_test.select("Outcome","rawPrediction","prediction","probability").show(5)
print("The area under ROC for train set is {}".format(evaluator.evaluate(predict_train)))
print("The area under ROC for test set is {}".format(evaluator.evaluate(predict_test)))


# In[44]:


# Hyperparameter tuning and K fold cross validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = ParamGridBuilder()    .addGrid(lr.aggregationDepth,[2,5,10])    .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])    .addGrid(lr.fitIntercept,[False, True])    .addGrid(lr.maxIter,[10, 100, 1000])    .addGrid(lr.regParam,[0.01, 0.5, 2.0])     .build()
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
cvModel = cv.fit(train)
predict_train=cvModel.transform(train)
predict_test=cvModel.transform(test)
print("The area under ROC for train set after CV  is {}".format(evaluator.evaluate(predict_train)))
print("The area under ROC for test set after CV  is {}".format(evaluator.evaluate(predict_test)))


# In[ ]:





# In[ ]:




