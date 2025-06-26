##Continuous Pipeline - NiFi Through Hbase

This project Is meant to show the flow of data from NiFi -> HDFS -> Hive -> PySpark -> Hbase.

Project Status: Completed

## Project Intro/Objective
The goal of this project is to take a data set and process it through a continuous data pipeline in a big data environment. The first step of the process was to load a dataset into HDFS from NiFi. Next, the data will be loaded into hive then pulled into pyspark  and have machine learning applied to it. Lastly, the metrics of machine learning were to be uploaded and retrieved from HBase. The following will discuss this process, what I tried and any technical issues I ran into.  

### Methods Used
* Predictive Modeling


### Technologies
* NiFi
* HDFS
* Hive
* PySpark
* Hbase


## Project Description
The Data that is used for this project is the Heart Attack data set for the predictive modeling portion. The Data set was shortened and modified from the original found on Kaggle outside of the environment. This is due to the project being completed in a training environment which couldn't handle the full data set.

# Challenges/ solutions:
* I ran the Invoke processor but the data kept flowing. Solution: Run the Invoke Processor one time only. This will load in the full data set.
* The PySpark session keeps crasing. Solution: This was due to no garbage collection being completed. Add a garbage collector after every couple of steps in the pyspark session to prevnt this issue.


## Needs of this project

- data exploration/descriptive statistics
- statistical modeling
- writeup/reporting


## Getting Started

1. The NiFi Flow that has the build for the processor group is the NiFi_Flow.json
2. The data that was used can be found in the data folder within this Repo. To Save stime from converting this to Json, I've included the medicaldata.json file in the main repo.
3. Screen shots for the pipeline can be found in the write up document or you can follow along from the full terminal text.
