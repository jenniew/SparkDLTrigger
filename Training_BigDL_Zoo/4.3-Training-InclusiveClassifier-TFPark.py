
# coding: utf-8

# Now we will train the Inclusive classifier, a combination of the Particle-sequence classifier with the High Level Features.
# 
# To run this notebook we used the following configuration:
# * *Software stack*: LCG 94 (it has spark 2.3.1)
# * *Platform*: centos7-gcc7
# * *Spark cluster*: Hadalytic

# In[1]:

# Check if Spark Session has been created correctly
spark


# In[2]:

# Add the BDL zip file
# sc.addPyFile("/eos/project/s/swan/public/BigDL/bigdl-0.7.0-python-api.zip")


# ## Load train and test dataset

# In[3]:




# ## Create the model

# In[5]:

# Init analytics zoo
from zoo.common.nncontext import *
from pyspark.sql import SQLContext

sc = init_nncontext("inclusive classifier")

PATH = "file:///data/cern/"

sql_context = SQLContext(sc)

trainDF = sql_context.read.format('parquet').load(PATH + 'trainUndersampled.parquet').select(
    ['GRU_input', 'HLF_input', 'encoded_label'])

testDF = sql_context.read.format('parquet').load(PATH + 'testUndersampled.parquet').select(
    ['GRU_input', 'HLF_input', 'encoded_label'])

# In[4]:

trainDF.printSchema()

# In[6]:

import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Masking, Dense, Activation, GRU, Dropout, concatenate
# from zoo.pipeline.api.keras.layers.torch import Select
# from zoo.pipeline.api.keras.layers mport BatchNormalization
# from zoo.pipeline.api.keras.layers import GRU
# from zoo.pipeline.api.keras.engine.topology import Merge

## GRU branch
gru_input = Input(shape=(801,19), name='gru_input')
masking = Masking(mask_value=0.)(gru_input)
gru = GRU(units=50,activation='tanh')(masking)
gruBranch = Dropout(0.2)(gru)
    
hlf_input = Input(shape=(14,), name='hlf_input')
hlfBranch = Dropout(0.2)(hlf_input)

concat = concatenate([gruBranch, hlfBranch])
dense = Dense(25, activation='relu')(concat)
output = Dense(3, activation='softmax')(dense)
    
model = Model(inputs=[gru_input, hlf_input], outputs=output)


# In[7]:

# from bigdl.util.common import Sample
import numpy as np

trainRDD = trainDF.rdd.map(lambda row: ([np.array(row.GRU_input), np.array(row.HLF_input)],
    np.array(row.encoded_label))
)

testRDD = testDF.rdd.map(lambda row: ([np.array(row.GRU_input), np.array(row.HLF_input)],
    np.array(row.encoded_label))
)


# trainRDD = trainDF.rdd.map(lambda row: Sample.from_ndarray(
#     [np.array(row.GRU_input), np.array(row.HLF_input)],
#     np.array(row.encoded_label)
# ))

# testRDD = testDF.rdd.map(lambda row: Sample.from_ndarray(
#     [np.array(row.GRU_input), np.array(row.HLF_input)],
#     np.array(row.encoded_label)
# ))


# In[8]:

# trainRDD.count()


# In[9]:

# testRDD.count()


# ## Create train and valiation Data
# 
# We need to create an RDD of a tuple of the form (`features`, `label`). The two elements of this touple should be `numpy arrays`. 

# In[10]:

# Let's have a look at one element of trainRDD
trainRDD.take(1)


# We can see that `features` is  now composed by the list of 801 particles with 19 features each (`shape=[801 19]`) plus the HLF (`shape=[14]`) and the encoded label (`shape=[3]`).

# In[11]:

from zoo.pipeline.api.net import TFDataset
from zoo.tfpark.model import KerasModel

# create TFDataset for TF training
dataset = TFDataset.from_rdd(trainRDD,
                                 features=[(tf.float32, [801, 19]), (tf.float32, [14])],
                                 labels=(tf.float32, [3]),
                                 batch_size=256,
                                 val_rdd=testRDD)


# ## Optimizer setup and training

# In[12]:

# Set of hyperparameters
numEpochs = 8

# The batch used by BDL must be a multiple of numExecutors * executorCores
# Because data will be equally distibuted inside each executor

workerBatch = 64
# numExecutors = int(sc._conf.get('spark.executor.instances'))
numExecutors = 1
# executorCores = int(sc._conf.get('spark.executor.cores'))
executorCores = 4

BDLbatch = workerBatch * numExecutors * executorCores


# In[13]:

# Use Keras model training API to train

from bigdl.optim.optimizer import *
# from bigdl.nn.criterion import CategoricalCrossEntropy

model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

keras_model = KerasModel(model)


# model.compile(optimizer='adam', loss=CategoricalCrossEntropy(), metrics=[Loss(CategoricalCrossEntropy())])


# Let's define a directory to store logs (i.e. train and validation losses) and save models

# In[14]:

# name of our application
appName = "InclusiveClassifier"

# Change it! 
logDir = "/data/cern/TFParklogs"

# Check if there is already an application with the same name 
# and remove it, otherwise logs will be appended to that app
import os
try:
    os.system('rm -rf '+logDir+'/'+appName)
except:
    pass

print("Saving logs to {}".format(logDir+'/'+appName))


# In[15]:

# Set tensorboard for model training and validation
# keras_model.set_tensorboard(logDir, appName)
trainSummary = TrainSummary(log_dir=logDir,app_name=appName)
valSummary = ValidationSummary(log_dir=logDir,app_name=appName)
keras_model.set_train_summary(trainSummary)
keras_model.set_val_summary(valSummary)


# In[ ]:

keras_model.fit(x=dataset, epochs=numEpochs, distributed=True)


# We are now ready to launch the training.
# 
# **Warnign: During the trainign it would be better to shutdown the Toggle Spark Monitorin Display because each iteration is seen as a spark job, therefore the toggle will try to display everything causing problem to the browser.** 

# In[55]:

# %%time 
# model.fit(x=trainRDD, batch_size=BDLbatch, nb_epoch=numEpochs, validation_data=testRDD, distributed=True)


# ## Plot loss

# In[57]:

# import matplotlib.pyplot as plt
# plt.style.use('seaborn-darkgrid')
# get_ipython().magic(u'matplotlib notebook')
#
# # trainSummary = TrainSummary(log_dir=logDir,app_name=appName)
# loss = np.array(trainSummary.read_scalar("Loss"))
# # valSummary = ValidationSummary(log_dir=logDir,app_name=appName)
# val_loss = np.array(valSummary.read_scalar("Loss"))
#
# plt.plot(loss[:,0], loss[:,1], label="Training loss")
# plt.plot(val_loss[:,0], val_loss[:,1], label="Validation loss", color='crimson', alpha=0.8)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.legend()
# plt.title("Particle sequence classifier loss")
# plt.show()
#

# ## Save the model

# In[59]:

modelDir = os.path.join(logDir, "models", "inclusive.model")
print modelDir
keras_model.save_model(modelDir)


# It is possible to load the model in the following way:
# ```Python
# model = Model.loadModel(modelPath=modelPath+'.bigdl', weightPath=modelPath+'.bin')
# ```

# ## Prediction

# In[60]:

testRDD2 = testDF.rdd.map(lambda row: [np.array(row.GRU_input), np.array(row.HLF_input)])
test_dataset = TFDataset.from_rdd(testRDD2,
                                 features=[(tf.float32, [801, 19]), (tf.float32, [14])],
                                 labels=None,
                                 batch_per_thread=64)


# In[61]:

predRDD = keras_model.predict(test_dataset)


# In[62]:

result = predRDD.collect()


# In[63]:

y_pred = np.squeeze(result)
y_true = np.asarray(testDF.select('encoded_label').rdd.map(lambda row: np.asarray(row.encoded_label)).collect())


# In[64]:

from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[65]:

# plt.figure()
# plt.plot(fpr[0], tpr[0], lw=2,
#          label='Inclusive classifier (AUC) = %0.4f' % roc_auc[0])
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Background Contamination (FPR)')
# plt.ylabel('Signal Efficiency (TPR)')
# plt.title('$tt$ selector')
# plt.legend(loc="lower right")
# plt.show()
#

# In[ ]:

loss = np.array(trainSummary.read_scalar("Accuracy"))
# valSummary = ValidationSummary(log_dir=logDir,app_name=appName)
val_loss = np.array(valSummary.read_scalar("Accuracy"))
# get_ipython().magic(u'matplotlib notebook')
# plt.figure()
#
# plt.plot(history.history['acc'], label='train')
# plt.plot(history.history['val_acc'], label='validation')
# plt.ylabel('Accuracy')
# plt.xlabel('epoch')
# plt.legend(loc='lower right')
# plt.title("HLF classifier accuracy")
# plt.show()


# In[ ]:




# In[ ]:



