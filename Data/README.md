## Data samples

This directory contains a few small samples of data. 
Data is stored using git lfs. You will need git lfs to download it, this is one way to configure git lfs (for CentOS):
```
yum install git-lfs
git lfs install
```

- [testUndersampled_2kevents.parquet](testUndersampled_2kevents.parquet) Contains a sample of the test dataset with all the features, in Apache Parquet format, produced by the filtering and feature engineering steps
- [rawData_sample](rawData_sample) Contains a sample of the raw data, one file per event topology type
- [trainUndersampled_HLF_features.parquet](trainUndersampled_HLF_features.parquet) High Level Features and labels in Apache Parquet format, training dataset
- [testUndersampled_HLF_features.parquet](testUndersampled_HLF_features.parquet) High Level Features and labels, in Apache Parquet format, test dataset
- [trainUndersampled_HLF_features.tfrecord](trainUndersampled_HLF_features.tfrecord) High Level Features and labels in TFRecord format, training dataset
- [testUndersampled_HLF_features.tfrecord](testUndersampled_HLF_features.tfrecord) High Level Features and labels, in TFRecord format, test dataset

The full dataset is expected to be released as CERN Open Data at a later date.
If you have access to CERN computing resources, you can contact the Hadoop and Spark service admins to get more information on how you can run this pipeline with the full dataset.

