****Key to understand and remember: Any question that asks anomaly detection, always go with Random Cut Forest (RCF) Algorithm.****



A machine learning specialist (ML) working for an investment firm is building a credit risk model to identify potentially risky loans. The training data includes 10,000 cases collected from past customers. Upon inspection, the specialist found that 95% of the cases consisted of fully repaid loans, and only 5% defaulted loans.

The company wants to minimize the number of loan rejections from non-defaulters.

Which method can the specialist implement to satisfy the requirements with the LEAST operational overhead?

* Run an Amazon SageMaker AutoPilot job using Area Under the ROC Curve (AUC) as the evaluation metric. Set the problem type to Binary classification.
* Ask for additional cases focusing on defaulted loans and add them to the training data. Use Amazon SageMaker’s built-in XGBoost algorithm with Area Under the ROC Curve (AUC) as the evaluation metric.
* Run an Amazon SageMaker AutoPilot job using accuracy as the evaluation metric. Set the problem type to Binary classification.
* Run an Amazon SageMaker AutoPilot job using accuracy as the evaluation metric. Set the problem type to Regression.

####################################################################


A Machine Learning Specialist is using Bayesian optimization to run an automatic hyperparameter tuning in SageMaker. He noticed that the tuning job takes a lot of compute resources and time to yield the best result, which drives a higher cost.

How can the Specialist reduce the cost of running the job?

* Reduce the number of concurrent tuning jobs.
* Use Amazon SageMaker Neo to accelerate the tuning job.
* Add the number of concurrent tuning jobs.
* Use Amazon Elastic Inference to accelerate the tuning job.

####################################################################

A Machine Learning Specialist has completed the training process of its binary classification model and is now ready to deploy the solution to the production environment. The model will diagnose an entire dataset on whether a given tumor is malignant or benign. The Specialist wants to remove the ID number attribute associated with each tumor before running the inference. However, the predicted results must contain both the tumor ID and its corresponding probability of being malignant.

Which tool should the Specialist use to deploy the model?

* Amazon SageMaker Hosting Services
* Amazon SageMaker Inference Pipeline
* Amazon SageMaker Batch Transform
* Amazon SageMaker Neo


####################################################################

A Machine Learning Specialist has trained a model using Amazon SageMaker built-in algorithms. The model exhibited a high prediction accuracy during experiments, but it generalized poorly when deployed in production.

Which action could the Specialist take to resolve the issue?

* Apply imputation techniques.
* Increase the learning rate.
* Add more features to the training data.
* Increase the applied regularization.

###################################################################

A Machine Learning Specialist is designing an Extract Transform Load (ETL) workflow that will run every day. The workflow consists of separate ETL jobs. The workflow is described as the following:

Initialize the workflow once the data is delivered to an S3 bucket.

Wait until all data is uploaded and available in Amazon S3. Then, execute an ETL job that will join newly uploaded data with multiple pre-existing data in Amazon S3.

Save the results to a separate S3 bucket.

The Specialist must be notified should one of the ETL jobs fail.

Which is the MOST efficient way of implementing the solution?

* Execute an AWS Step Functions workflow using a Lambda function. Join the uploaded and pre-existing datasets with AWS Glue and persist the results in a separate S3 bucket. Use Amazon CloudWatch Alarm to publish notifications to an SNS topic in case one of the ETL jobs fails.
* Use AWS Lambda to launch an Amazon SageMaker instance. Write a lifecycle configuration script for joining the datasets. Store the results in a separate S3 bucket. Use Amazon CloudWatch Alarm to publish notifications to an SNS topic in case one of the ETL jobs fails.
* Implement the ETL workflow using AWS Batch and trigger it when data is uploaded into an S3 bucket. Join the uploaded and pre-existing datasets with AWS Glue and persist the results in a separate S3 bucket. Use Amazon CloudWatch Alarm to publish notifications to an SNS topic in case one of the ETL jobs fails.
* Trigger a chain of Lambda functions that will join the datasets when data is uploaded to an S3 bucket. Use Amazon CloudWatch Alarm to publish notifications to an SNS topic in case one of the ETL jobs fails.

###################################################################

A Machine Learning Specialist is developing an image classification model in Amazon SageMaker using 25 epochs. During training, he has observed that the validation loss starts to increase from the 15th epoch onwards. This results in poor model performance with expensive and slow training times.

Which method can the Specialist do to prevent this issue from happening in the future?

* Enable the “Early Stopping” option
* Perform Principal Component Analysis
* Run data augmentation techniques on the image data before training
* Reduce regularization

###################################################################

A Machine Learning Specialist will be using different sets of training data to evaluate the various machine learning classification models against each other. She wants to rank each generated model by its ability to predict true positives.

Which performance metric is the MOST appropriate for the problem?

* Mean Absolute Percentage Error(MAPE)
* Root-mean-square error (RMSE)
* Specificity
* Area Under the ROC Curve (AUC)

###################################################################

A Machine Learning Specialist wants to build a click prediction model using an Amazon SageMaker built-in algorithm. The model will capture the click-through rate (CTR) pattern on banner ads of varying sizes located on different web pages.

Which built-in algorithm is the MOST suitable for this problem?

* Factorization Machines
* Principal Component Analysis (PCA)
* Latent Dirichlet Allocation (LDA)
* DeepAR Forecasting

###################################################################

A Machine Learning Specialist has collected a large training dataset of different cat breeds. The Specialist intends to use this for building a neural network model that can identify the breed of a given cat image. The Specialist wants to take advantage of a pre-trained model for his project through transfer learning.

How should the Specialist re-train the network with his own training data?

* Initialize the network with pre-trained weights in all layers.
* Initialize the network with pre-trained weights in all layers except for the output layer. Initialize the output layer with random weights.
* Initialize the network with random weights in all layers except for the output layer. Initialize the output layer with pre-trained weights.
* Retrain the network from scratch using the collected image data.

###################################################################

https://github.com/Ditectrev/Amazon-Web-Services-Certified-AWS-Certified-Machine-Learning-MLS-C01-Practice-Tests-Exams-Question/blob/main/README.md


### A company needs to quickly make sense of a large amount of data and gain insight from it. The data is in different formats, the schemas change frequently, and new data sources are added regularly. The company wants to use AWS services to explore multiple data sources, suggest schemas, and enrich and transform the data. The solution should require the least possible coding effort for the data flows and the least possible infrastructure management. Which combination of AWS services will meet these requirements?

- [ ] Amazon EMR for data discovery, enrichment, and transformation. Amazon Athena for querying and analyzing the results in Amazon S3 using standard SQL. Amazon QuickSight for reporting and getting insights.
- [ ] Amazon Kinesis Data Analytics for data ingestion. Amazon EMR for data discovery, enrichment, and transformation. Amazon Redshift for querying and analyzing the results in Amazon S3.
- [x] AWS Glue for data discovery, enrichment, and transformation. Amazon Athena for querying and analyzing the results in Amazon S3 using standard SQL. Amazon QuickSight for reporting and getting insights.
- [ ] AWS Data Pipeline for data transfer. AWS Step Functions for orchestrating AWS Lambda jobs for data discovery, enrichment, and transformation. Amazon Athena for querying and analyzing the results in Amazon S3 using standard SQL. Amazon QuickSight for reporting and getting insights.

### A financial company is trying to detect credit card fraud. The company observed that, on average, 2% of credit card transactions were fraudulent. A data scientist trained a classifier on a year's worth of credit card transactions data. The model needs to identify the fraudulent transactions (positives) from the regular ones (negatives). The company's goal is to accurately capture as many positives as possible. Which metrics should the data scientist use to optimize the model? (Choose two.)

- [ ] Specificity.
- [ ] False positive rate.
- [ ] Accuracy.
- [x] Area under the precision-recall curve.
- [x] True positive rate.


### A data scientist needs to identify fraudulent user accounts for a company's ecommerce platform. The company wants the ability to determine if a newly created account is associated with a previously known fraudulent user. The data scientist is using AWS Glue to cleanse the company's application logs during ingestion. Which strategy will allow the data scientist to identify fraudulent accounts?

- [ ] Execute the built-in FindDuplicates Amazon Athena query.
- [x] Create a FindMatches machine learning transform in AWS Glue.
- [ ] Create an AWS Glue crawler to infer duplicate accounts in the source data.
- [ ] Search for duplicate accounts in the AWS Glue Data Catalog.


### A data scientist has developed a machine learning translation model for English to Japanese by using Amazon SageMaker's built-in seq2seq algorithm with 500,000 aligned sentence pairs. While testing with sample sentences, the data scientist finds that the translation quality is reasonable for an example as short as five words. However, the quality becomes unacceptable if the sentence is 100 words long. Which action will resolve the problem?

- [ ] Change preprocessing to use n-grams.
- [ ] Add more nodes to the Recurrent Neural Network (RNN) than the largest sentence's word count.
- [x] Adjust hyperparameters related to the attention mechanism.
- [ ] Choose a different weight initialization type.

### An e commerce company wants to launch a new cloud-based product recommendation feature for its web application. Due to data localization regulations, any sensitive data must not leave its on-premises data center, and the product recommendation model must be trained and tested using nonsensitive data only. Data transfer to the cloud must use IPsec. The web application is hosted on premises with a PostgreSQL database that contains all the data. The company wants the data to be uploaded securely to Amazon S3 each day for model retraining. How should a machine learning specialist meet these requirements?

- [x] Create an AWS Glue job to connect to the PostgreSQL DB instance. Ingest tables without sensitive data through an AWS Site-to-Site VPN connection directly into Amazon S3.
- [ ] Create an AWS Glue job to connect to the PostgreSQL DB instance. Ingest all data through an AWS Site-to-Site VPN connection into Amazon S3 while removing sensitive data using a PySpark job.
- [ ] Use AWS Database Migration Service (AWS DMS) with table mapping to select PostgreSQL tables with no sensitive data through an SSL connection. Replicate data directly into Amazon S3.
- [ ] Use PostgreSQL logical replication to replicate all data to PostgreSQL in Amazon EC2 through AWS Direct Connect with a VPN connection. Use AWS Glue to move data from Amazon EC2 to Amazon S3.

### The chief editor for a product catalog wants the research and development team to build a machine learning system that can be used to detect whether or not individuals in a collection of images are wearing the company's retail brand. The team has a set of training data. Which machine learning algorithm should the researchers use that BEST meets their requirements?

- [ ] Latent Dirichlet Allocation (LDA).
- [ ] Recurrent Neural Network (RNN).
- [ ] K-means.
- [x] Convolutional Neural Network (CNN).

### A Machine Learning Specialist is designing a scalable data storage solution for Amazon SageMaker. There is an existing TensorFlow-based model implemented as a train.py script that relies on static training data that is currently stored as TFRecords. Which method of providing training data to Amazon SageMaker would meet the business requirements with the LEAST development overhead?

- [ ] Use Amazon SageMaker script mode and use train.py unchanged. Point the Amazon SageMaker training invocation to the local path of the data without reformatting the training data.
- [x] Use Amazon SageMaker script mode and use train.py unchanged. Put the TFRecord data into an Amazon S3 bucket. Point the Amazon SageMaker training invocation to the S3 bucket without reformatting the training data.
- [ ] Rewrite the train.py script to add a section that converts TFRecords to protobuf and ingests the protobuf data instead of TFRecords.
- [ ] Prepare the data in the format accepted by Amazon SageMaker. Use AWS Glue or AWS Lambda to reformat and store the data in an Amazon S3 bucket.

### A manufacturer is operating a large number of factories with a complex supply chain relationship where unexpected downtime of a machine can cause production to stop at several factories. A data scientist wants to analyze sensor data from the factories to identify equipment in need of preemptive maintenance and then dispatch a service team to prevent unplanned downtime. The sensor readings from a single machine can include up to 200 data points including temperatures, voltages, vibrations, RPMs, and pressure readings. To collect this sensor data, the manufacturer deployed Wi-Fi and LANs across the factories. Even though many factory locations do not have reliable or high- speed internet connectivity, the manufacturer would like to maintain near-real-time inference capabilities. Which deployment architecture for the model will address these business requirements?

- [ ] Deploy the model in Amazon SageMaker. Run sensor data through this model to predict which machines need maintenance.
- [x] Deploy the model on AWS IoT Greengrass in each factory. Run sensor data through this model to infer which machines need maintenance.
- [ ] Deploy the model to an Amazon SageMaker batch transformation job. Generate inferences in a daily batch report to identify machines that need maintenance.
- [ ] Deploy the model in Amazon SageMaker and use an IoT rule to write data to an Amazon DynamoDB table. Consume a DynamoDB stream from the table with an AWS Lambda function to invoke the endpoint.


### A data scientist has explored and sanitized a dataset in preparation for the modeling phase of a supervised learning task. The statistical dispersion can vary widely between features, sometimes by several orders of magnitude. Before moving on to the modeling phase, the data scientist wants to ensure that the prediction performance on the production data is as accurate as possible. Which sequence of steps should the data scientist take to meet these requirements?

- [ ] Apply random sampling to the dataset. Then split the dataset into training, validation, and test sets.
- [x] Split the dataset into training, validation, and test sets. Then rescale the training set and apply the same scaling to the validation and test sets.
- [ ] Rescale the dataset. Then split the dataset into training, validation, and test sets.
- [ ] Split the dataset into training, validation, and test sets. Then rescale the training set, the validation set, and the test set independently.

### A Machine Learning Specialist working for an online fashion company wants to build a data ingestion solution for the company's Amazon S3-based data lake. The Specialist wants to create a set of ingestion mechanisms that will enable future capabilities comprised of: Real-time analytics. Interactive analytics of historical data. Clickstream analytics. Product recommendations. Which services should the Specialist use?

- [x] AWS Glue as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for real-time data insights; Amazon Kinesis Data Firehose
for delivery to Amazon ES for clickstream analytics; Amazon EMR to generate personalized product recommendations.
- [ ] Amazon Athena as the data catalog: Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for near-real-time data insights; Amazon Kinesis
Data Firehose for clickstream analytics; AWS Glue to generate personalized product recommendations.
- [ ] AWS Glue as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for historical data insights; Amazon Kinesis Data Firehose
for delivery to Amazon ES for clickstream analytics; Amazon EMR to generate personalized product recommendations.
- [ ] Amazon Athena as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for historical data insights; Amazon DynamoDB streams for clickstream analytics; AWS Glue to generate personalized product recommendations.


### A Machine Learning Specialist built an image classification deep learning model. However, the Specialist ran into an overfitting problem in which the training and testing accuracies were 99% and 75%, respectively. How should the Specialist address this issue and what is the reason behind it?

- [ ] The learning rate should be increased because the optimization process was trapped at a local minimum.
- [x] The dropout rate at the flatten layer should be increased because the model is not generalized enough.
- [ ] The dimensionality of dense layer next to the flatten layer should be increased because the model is not complex enough.
- [ ] The epoch number should be increased because the optimization process was terminated before it reached the global minimum.


### A Machine Learning Specialist is building a prediction model for a large number of features using linear models, such as Linear Regression and Logistic Regression. During exploratory data analysis, the Specialist observes that many features are highly correlated with each other. This may make the model unstable. What should be done to reduce the impact of having such a large number of features?

- [ ] Perform one-hot encoding on highly correlated features.
- [ ] Use matrix multiplication on highly correlated features.
- [x] Create a new feature space using Principal Component Analysis (PCA).
- [ ] Apply the Pearson correlation coefficient.


### A Data Scientist needs to create a serverless ingestion and analytics solution for high-velocity, real-time streaming data. The ingestion process must buffer and convert incoming records from JSON to a query-optimized, columnar format without data loss. The output datastore must be highly available, and Analysts must be able to run SQL queries against the data and connect to existing business intelligence dashboards. Which solution should the Data Scientist build to satisfy the requirements?

- [x] Create a schema in the AWS Glue Data Catalog of the incoming data format. Use an Amazon Kinesis Data Firehose delivery stream to stream the data and transform the data to Apache Parquet or ORC format using the AWS GlueData Catalog before delivering to Amazon S3. Have the Analysts query the data directly from Amazon S3 using Amazon Athena, and connect to BI tools using the Athena Java Database Connectivity (JDBC) connector.
- [ ] Write each JSON record to a staging location in Amazon S3. Use the S3 Put event to trigger an AWS Lambda function that transforms the data into Apache Parquet or ORC format and writes the data to a processed data location inAmazon S3. Have the Analysts query the data directly from Amazon S3 using Amazon Athena, and connect to BI tools using the Athena Java Database Connectivity (JDBC) connector.
- [ ] Write each JSON record to a staging location in Amazon S3. Use the S3 Put event to trigger an AWS Lambda function that transforms the data into Apache Parquet or ORC format and inserts it into an Amazon RDS PostgreSQLdatabase. Have the Analysts query and run dashboards from the RDS database.
- [ ] Use Amazon Kinesis Data Analytics to ingest the streaming data and perform real-time SQL queries to convert the records to Apache Parquet before delivering to Amazon S3. Have the Analysts query the data directly from Amazon S3using Amazon Athena and connect to BI tools using the Athena Java Database Connectivity (JDBC) connector.

### A manufacturing company has a large set of labeled historical sales data. The manufacturer would like to predict how many units of a particular part should be produced each quarter. Which machine learning approach should be used to solve this problem?

- [ ] Logistic Regression.
- [ ] Random Cut Forest (RCF).
- [ ] Principal Component Analysis (PCA).
- [x] Linear Regression.

### A Machine Learning Specialist is assigned a TensorFlow project using Amazon SageMaker for training, and needs to continue working for an extended period with no Wi-Fi access. Which approach should the Specialist use to continue working?

- [ ] Install Python 3 and boto3 on their laptop and continue the code development using that environment.
- [x] Download the TensorFlow Docker container used in Amazon SageMaker from GitHub to their local environment, and use the Amazon SageMaker Python SDK to test the code.
- [ ] Download TensorFlow from tensorflow.org to emulate the TensorFlow kernel in the SageMaker environment.
- [ ] Download the SageMaker notebook to their local environment, then install Jupyter Notebooks on their laptop and continue the development in a local notebook.

### A retail company intends to use machine learning to categorize new products. A labeled dataset of current products was provided to the Data Science team. The dataset includes 1,200 products. The labeled dataset has 15 features for each product such as title dimensions, weight, and price. Each product is labeled as belonging to one of six categories such as books, games, electronics, and movies. Which model should be used for categorizing new products using the provided dataset for training?

- [x] AnXGBoost model where the objective parameter is set to multi:softmax.
- [ ] A deep Convolutional Neural Network (CNN) with a softmax activation function for the last layer.
- [ ] A regression forest where the number of trees is set equal to the number of product categories.
- [ ] A DeepAR forecasting model based on a Recurrent Neural Network (RNN).

### A Machine Learning Specialist uploads a dataset to an Amazon S3 bucket protected with server-side encryption using AWS KMS. How should the ML Specialist define the Amazon SageMaker notebook instance so it can read the same dataset from Amazon S3?

- [ ] Define security group(s) to allow all HTTP inbound/outbound traffic and assign those security group(s) to the Amazon SageMaker notebook instance.
- [ ] Configure the Amazon SageMaker notebook instance to have access to the VPC. Grant permission in the KMS key policy to the notebook's KMS role.
- [x] Assign an IAM role to the Amazon SageMaker notebook with S3 read access to the dataset. Grant permission in the KMS key policy to that role.
- [ ] Assign the same KMS key used to encrypt data in Amazon S3 to the Amazon SageMaker notebook instance.

### A large consumer goods manufacturer has the following products on sale: 34 different toothpaste variants. 48 different toothbrush variants. 43 different mouthwash variants. The entire sales history of all these products is available in Amazon S3. Currently, the company is using custom-built autoregressive integrated moving average (ARIMA) models to forecast demand for these products. The company wants to predict the demand for a new product that will soon be launched. Which solution should a Machine Learning Specialist apply?

- [ ] Train a custom ARIMA model to forecast demand for the new product.
- [x] Train an Amazon SageMaker DeepAR algorithm to forecast demand for the new product.
- [ ] Train an Amazon SageMaker K-means clustering algorithm to forecast demand for the new product.
- [ ] Train a custom XGBoost model to forecast demand for the new product.

### An agency collects census information within a country to determine healthcare and social program needs by province and city. The census form collects responses for approximately 500 questions from each citizen. Which combination of algorithms would provide the appropriate insights? (Select TWO.)

- [ ] The factorization machines (FM) algorithm.
- [ ] The Latent Dirichlet Allocation (LDA) algorithm.
- [x] The Principal Component Analysis (PCA) algorithm.
- [x] The K-means algorithm.
- [ ] The Random Cut Forest (RCF) algorithm.

### An aircraft engine manufacturing company is measuring 200 performance metrics in a time-series. Engineers want to detect critical manufacturing defects in near-real time during testing. All of the data needs to be stored for offline analysis. What approach would be the MOST effective to perform near-real time defect detection?

- [ ] Use AWS IoT Analytics for ingestion, storage, and further analysis. Use Jupyter notebooks from within AWS IoT Analytics to carry out analysis for anomalies.
- [ ] Use Amazon S3 for ingestion, storage, and further analysis. Use an Amazon EMR cluster to carry out Apache Spark ML K-means clustering to determine anomalies.
- [ ] Use Amazon S3 for ingestion, storage, and further analysis. Use the Amazon SageMaker Random Cut Forest (RCF) algorithm to determine anomalies.
- [x] Use Amazon Kinesis Data Firehose for ingestion and Amazon Kinesis Data Analytics Random Cut Forest (RCF) to perform anomaly detection. Use Kinesis Data Firehose to store data in Amazon S3 for further analysis.


### A Data Scientist is building a model to predict customer churn using a dataset of 100 continuous numerical features. The Marketing team has not provided any insight about which features are relevant for churn prediction. The Marketing team wants to interpret the model and see the direct impact of relevant features on the model outcome. While training a Logistic Regression model, the Data Scientist observes that there is a wide gap between the training and validation set accuracy. Which methods can the Data Scientist use to improve the model performance and satisfy the Marketing team's needs? (Choose two.)

- [x] Add L1 regularization to the classifier.
- [ ] Add features to the dataset.
- [x] Perform recursive feature elimination.
- [ ] Perform t-distributed stochastic neighbor embedding (t-SNE).
- [ ] Perform linear discriminant analysis.

### A company uses a long short-term memory (LSTM) model to evaluate the risk factors of a particular energy sector. The model reviews multi-page text documents to analyze each sentence of the text and categorize it as either a potential risk or no risk. The model is not performing well, even though the Data Scientist has experimented with many different network structures and tuned the corresponding hyperparameters. Which approach will provide the MAXIMUM performance boost?

- [ ] Initialize the words by term frequency-inverse document frequency (TF-IDF) vectors pretrained on a large collection of news articles related to the energy sector.
- [ ] Use gated recurrent units (GRUs) instead of LSTM and run the training process until the validation loss stops decreasing.
- [ ] Reduce the learning rate and run the training process until the training loss stops decreasing.
- [x] Initialize the words by word2vec embeddings pretrained on a large collection of news articles related to the energy sector.


### A web-based company wants to improve its conversion rate on its landing page. Using a large historical dataset of customer visits, the company has repeatedly trained a multi-class deep learning network algorithm on Amazon SageMaker. However, there is an overfitting problem: training data shows 90% accuracy in predictions, while test data shows 70% accuracy only. The company needs to boost the generalization of its model before deploying it into production to maximize conversions of visits to purchases. Which action is recommended to provide the HIGHEST accuracy model for the company's test and validation data?

- [ ] Increase the randomization of training data in the mini-batches used in training.
- [ ] Allocate a higher proportion of the overall data to the training dataset.
- [x] Apply L1 or L2 regularization and dropouts to the training.
- [ ] Reduce the number of layers and units (or neurons) from the deep learning network.

### A Data Scientist needs to analyze employment data. The dataset contains approximately 10 million observations on people across 10 different features. During the preliminary analysis, the Data Scientist notices that income and age distributions are not normal. While income levels shows a right skew as expected, with fewer individuals having a higher income, the age distribution also shows a right skew, with fewer older individuals participating in the workforce. Which feature transformations can the Data Scientist apply to fix the incorrectly skewed data? (Choose two.)

- [ ] Cross-validation.
- [x] Numerical value binning.
- [ ] High-degree polynomial transformation.
- [x] Logarithmic transformation.
- [ ] One hot encoding.







