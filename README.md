# SparkDLTrigger

This repository contains the code and notebooks accompanying the blog article
[Machine Learning Pipelines for High Energy Physics Using Apache Spark with BigDL and Analytics Zoo](https://db-blog.web.cern.ch/blog/luca-canali/machine-learning-pipelines-high-energy-physics-using-apache-spark-bigdl).    

Principal author of the notebooks: Matteo.Migliorini@cern.ch  
Further work and contacts: Luca.Canali@cern.ch; Riccardo.Castellotti@cern.ch; Matteo.Migliorini@cern.ch    
Raw data and neural network models: [T.Q. Nguyen *et al.*](https://arxiv.org/abs/1807.00083)   
Acknowledgements: Viktor Khristenko, Thong Nguyen, Maurizio Pierini, Maria Girone, Marco Zanetti, 
members of the Hadoop and Spark service at CERN,
Intel team for BigDL and Analytics Zoo consultancy: Jiao (Jennie) Wang and Sajan Govindan,  
  
## Physics use case
Event data flows collected from the particle detector (CMS experiment) contains different types
of event topologies of interest. 
A particle classifier built with neural networks can be used as event filter,
improving state of the art in accuracy.  
This work reproduces the findings of the paper
[Topology classification with deep learning to improve real-time event selection at the LHC](https://arxiv.org/abs/1807.00083)
using tools from the Big Data ecosystem, notably Apache Spark and BigDL/Analytics Zoo.

![Physics use case for the particle classifier](Docs/Physics_use_case.png)
  
  
## Data pipelines for deeplearning
Data pipelines are of paramount importance to make machine learning projects successful, by integrating multiple components and APIs used for data processing across the entire data chain. A good data pipeline implementation can accelerate and improve the productivity of the work around the core machine learning tasks.
The four steps of the pipeline we built are:

- Data Ingestion: where we read data from ROOT format and from the CERN-EOS storage system, into a Spark DataFrame and save the results as a table stored in Apache Parquet files
- Feature Engineering and Event Selection: where the Parquet files containing all the events details processed in Data Ingestion are filtered and datasets with new  features are produced
- Parameter Tuning: where the best set of hyperparameters for each model architecture are found performing a grid search
- Training: where the best models found in the previous step are trained on the entire dataset.

![Machine learning data pipeline](Docs/DataPipeline.png)
  
## Results
The results of the DL model(s) training are satisfactoy and match the results of the original research paper. 
![Loss converging, ROC and AUC](Docs/Loss_ROC_AUC.png)

## Data and code
Data and code to reproduce this work are made available via this repository.
- [Links to download the datasets](Data), with a short description
- Notebooks with [data preparation code using Apache Spark](DataIngestion_FeaturePreparation)
- Notebooks with machine learning training
  - Distributed DL training with [Apache Spark and BigDL/AnalyticsZoo](Training_BigDL_Zoo)
  - Training DL models with TensorFlow (tf.keras):
    - [tf.keras on CPU](Training_TFKeras_CPU), multiple methods
    - [Distributed Tensorflow on CPU](Training_TFKeras_CPU_Distributed), with tf.keras with tf.distributed
    - [tf.keras with GPU](Training_TFKeras_GPU)
    - [Saved models](Models)
  - Notebooks for [training with Spark ML](Training_Other_ML)  

## Additional info and references
- [Blog post "Machine Learning Pipelines for High Energy Physics Using Apache Spark with BigDL and Analytics Zoo"](https://db-blog.web.cern.ch/blog/luca-canali/machine-learning-pipelines-high-energy-physics-using-apache-spark-bigdl)
- [Poster at the CERN openlab technical workshop 2019](Docs/Poster.pdf)  
- [Presentation at Spark Summit SF 2019](https://databricks.com/session/deep-learning-on-apache-spark-at-cerns-large-hadron-collider-with-intel-technologies)  
  