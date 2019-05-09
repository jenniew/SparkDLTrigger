# SparkDLTrigger

The goal of this work is to reproduce the results of this paper [https://arxiv.org/abs/1807.00083](https://arxiv.org/abs/1807.00083)
using tools from the Big Data ecosystem.    
Principal author: Matteo Migliorini  
Contacts: Luca.Canali@cern.ch; Matteo.Migliorini@cern.ch; Riccardo.Castellotti@cern.ch  
Acknowledgements: Viktor Khristenko, Maria Girone, Maurizio Pierini, Thong Nhuyen, Members of the Hadoop and Spark service at CERN  
Intel for BigDL and Analytics Zoo consultancy: Jiao Wang and Sajan Govindan  

  
### Physics use case
Event data flows collected from the particle detector (CMS experiment) contains different types
of event topologies of interest. 
A particle classifier built with neural networks can be used as event filter,
improving state of the art in accuracy.
![Physics use case for the particle classifier](Docs/Physics_use_case.png)
  
  
### Data pipeline
Data pipelines are of paramount importance to make machine learning projects successful, by integrating multiple components and APIs used for data processing across the entire data chain. A good data pipeline implementation can accelerate and improve the productivity of the work around the core machine learning tasks.
The four steps of the pipeline we built are:

- Data Ingestion: where we read data from ROOT format and from the CERN-EOS storage system, into a Spark DataFrame and save the results as a table stored in Apache Parquet files
- Feature Engineering and Event Selection: where the Parquet files containing all the events details processed in Data Ingestion are filtered and datasets with new  features are produced
- Parameter Tuning: where the best set of hyperparameters for each model architecture are found performing a grid search
- Training: where the best models found in the previous step are trained on the entire dataset.

![Machine learning data pipeline](Docs/DataPipeline.png)
  
 
### Results
The resutls of the DL model(s) training are satisfactoy and match the resutls of the original research paper. 
![Loss converging, ROC and AUC](Docs/Loss_ROC_AUC.png)

### Additional info
- [Blog post "Machine Learning Pipelines for High Energy Physics Using Apache Spark with BigDL and Analytics Zoo"](https://db-blog.web.cern.ch/blog/luca-canali/machine-learning-pipelines-high-energy-physics-using-apache-spark-bigdl)
- [Poster at the CERN openlab technical workshop 2019](Docs/Poster.pdf)  
  
  
