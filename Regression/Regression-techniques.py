#!/usr/bin/env python
# coding: utf-8

# # Regression techniques using apache spark mllib

# In[1]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()


# In[ ]:





# In[ ]:





# ## Data reading and preprocessing

# In[2]:


# reading data from csv file locally
df = spark.read.format("csv").option("header", "true").load("linear_reg.csv")


# In[6]:


# viewing dataframe
df.show(10)


# In[ ]:





# In[35]:


# caching our dataframe
df.cache()


# In[70]:


# changing columns datatype to int 
df = df.withColumn("x", df["x"].cast('int')).withColumn("y", df["y"].cast('int'))


# In[ ]:





# In[ ]:





# ## Plotting

# In[71]:


# sampling and plotting the dataframe
import pandas as pd
import matplotlib.pyplot as plt
sampled_data = df.select('x','y').sample(False, 0.8).toPandas()
plt.scatter(sampled_data.x,sampled_data.y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('relation of y and x')
plt.show()


# In[74]:


# check correlation
df.stat.corr('x','y')


# In[75]:


from pyspark.ml.feature import VectorAssembler


# In[78]:


vectorAssembler = VectorAssembler(inputCols = ['x'], outputCol = 'features')


# In[79]:


v_df = vectorAssembler.transform(df)
v_df = v_df.select(['features', 'y'])
v_df.show(3)


# In[ ]:





# In[ ]:





# ## Implementing linear regression

# In[84]:


from pyspark.ml.regression import LinearRegression


# In[81]:


splits = v_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]


# In[92]:


# creating object
lr = LinearRegression(featuresCol = 'features', labelCol='y', maxIter=100, regParam=0.3, elasticNetParam=0.8)


# In[93]:


# fitting training data
lr_model = lr.fit(train_df)


# In[94]:


# coefficients and intercepts
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))


# In[95]:


# training summary
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# In[89]:


# testing details
lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","y","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="y",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))


# In[97]:


# root mean squared error on test data
test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)


# In[ ]:





# In[ ]:





# ## Implementing Decision Tree Regressor

# In[103]:


# creating object and fitting
from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'y')
dt_model = dt.fit(train_df)


# In[104]:


# predicting 
dt_predictions = dt_model.transform(test_df)
dt_predictions.show()


# In[ ]:


# evaluating
dt_evaluator = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


dt_evaluator1 = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="r2")
r2 = dt_evaluator1.evaluate(dt_predictions)
print("r2 on test data = %g" % r2)


# In[ ]:





# In[ ]:





# ## Implementing Gradient-boosted tree regression

# In[108]:


# creating object, fitting and predicting
from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'y', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)
gbt_predictions.select('prediction', 'y', 'features').show(5)


# In[110]:


# evaluating 
gbt_evaluator = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="rmse")
rmse = gbt_evaluator.evaluate(gbt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

gbt_evaluator1 = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="r2")
r2 = gbt_evaluator1.evaluate(gbt_predictions)
print("R2 on test data = %g" % r2)


# In[ ]:





# In[ ]:





# ## Implementing Random forest regression

# In[112]:


# creating object, fitting and predicting
from pyspark.ml.regression import RandomForestRegressor
rf = RandomForestRegressor(featuresCol="features",labelCol = 'y',maxDepth=5)
rf_model = gbt.fit(train_df)
rf_predictions = rf_model.transform(test_df)
rf_predictions.select('prediction', 'y', 'features').show(5)


# In[113]:


# evaluating 
rf_evaluator = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="rmse")
rmse = rf_evaluator.evaluate(rf_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

rf_evaluator1 = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="r2")
r2 = rf_evaluator1.evaluate(rf_predictions)
print("R2 on test data = %g" % r2)


# In[ ]:





# In[ ]:





# ## Conclusion
# ### Linear regresion is woking very well among each regrassor with r2=1.

# In[ ]:




