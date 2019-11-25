#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()


# In[105]:


# reading data from csv file locally
df = spark.read.format("csv").option("header", "true").load("income_and_expenditure.csv")


# In[106]:


df.count()


# In[107]:


df.show(5)


# In[108]:


df.printSchema()


# In[109]:


# changing columns datatype to int 
df = df.withColumn("INCOME", df["INCOME"].cast('int')).withColumn("SPEND", df["SPEND"].cast('int'))


# In[110]:


df.printSchema()


# In[111]:


# checking for null values
df.createOrReplaceTempView('temp_df')
my_df = spark.sql("select * from temp_df where INCOME=NULL or SPEND=NULL")
my_df.show()


# In[112]:


# sampling and plotting the dataframe
import pandas as pd
import matplotlib.pyplot as plt
x='INCOME'
y='SPEND'
sampled_data = df.select(x,y).sample(False, 0.8).toPandas()
plt.scatter(sampled_data.INCOME,sampled_data.SPEND)
plt.xlabel(x)
plt.ylabel(y)
plt.title('relation of INCOME and SPEND')
plt.show()


# In[113]:


# assembling the columns
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = [x,y], outputCol = 'features')
v_df = vectorAssembler.transform(df)
v_df = v_df.select(['features'])
v_df.show(3)


# In[90]:


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# In[114]:


# creating objects with four centers and fitting
kmeans = KMeans().setK(4).setSeed(1)
model = kmeans.fit(v_df)


# In[115]:


# predicting centers
predictions = model.transform(v_df)


# In[116]:


predictions.show(5)


# In[ ]:





# In[117]:


# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))


# In[118]:


# Shows the center points
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[119]:


predictions.select('features').show(5)


# In[97]:


from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType


# In[120]:



split1_udf = udf(lambda value: value[0].item(), FloatType())
split2_udf = udf(lambda value: value[1].item(), FloatType())


# In[121]:


# unpacking features column so that we can plot them 
predictions=predictions.withColumn('x', split1_udf('features')).withColumn('y', split2_udf('features'))


# In[122]:


predictions.show(4)


# In[123]:


# sampling data to plot 
data = predictions.select('x','y','prediction').sample(False, 0.8).toPandas()


# In[124]:


x_cord = [[],[],[],[]]
y_cord = [[],[],[],[]]


# In[128]:


cx = []
cy = []
for center in model.clusterCenters():
    cx.append(center[0])
    cy.append(center[1])


# In[129]:


for i in range(len(data)):
    x_cord[data.loc[i,'prediction']].append(data.loc[i,'x'])
    y_cord[data.loc[i,'prediction']].append(data.loc[i,'y'])


# In[133]:


# plotting the clusters
plt.scatter(x_cord[0],y_cord[0],c='yellow')
plt.scatter(x_cord[1],y_cord[1],c='orange')
plt.scatter(x_cord[2],y_cord[2],c='cyan')
plt.scatter(x_cord[3],y_cord[3],c='magenta')
plt.scatter(cx,cy,c='blue',marker='x')
plt.xlabel('Income')
plt.ylabel('Expenditure')
plt.title('KMeans Result')
plt.show()


# ### You can clearly see four clusters with centers in blue colors

# In[ ]:




