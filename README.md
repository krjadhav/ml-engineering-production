# Machine Learning Engineering for Production (MLOps) Specialization

This README is a summary of what was learned from completing the Machine Learning Engineering for Production (MLOps) Specialization. The course provides a deep understanding of how to design and implement end-to-end machine learning production pipelines.

## Course 1: Machine Learning Data Lifecycle in Production

### Key Concepts

- Understanding the data validation steps
- Utilization of TensorFlow Extended (TFX) for feature engineering
- Building a production data pipeline

### Projects

1. **Data Validation**: Examining the characteristics of the Diabetes dataset including generating and visualizing statistics, inferring and updating a dataset schema, detecting, visualizing, and fixing anomalies, and examining slices of the dataset.

2. **Feature Engineering**: Building a data pipeline using TFX to prepare features from the Metro Interstate Traffic Volume dataset. This includes creating an InteractiveContext, using TFX ExampleGen component, generating the statistics and the schema, validating the evaluation dataset statistics, and performing feature engineering.

3. **Data Pipeline Components for Production ML**: Handling the first three steps of a production machine learning project - Data ingestion, Data Validation, and Data Transformation.

---

## Course 2: Machine Learning Modeling Pipelines in Production

### Key Concepts

- Training custom machine learning models
- Understanding distributed training using Kubernetes
- Deploying ML models with Vertex AI

### Projects

1. **Classify Images of Clouds in the Cloud with AutoML Vision**: Training a model on an image dataset, which includes uploading training images, training a model in the AutoML UI, and generating predictions on new cloud images.

2. **Distributed Multi-worker TensorFlow Training on Kubernetes**: Scaling out TensorFlow distributed training using Google Cloud Kubernetes Engine (GKE) and Kubeflow TFJob. This includes deploying TFJob components to GKE, configuring multi-worker distributed training jobs, and submitting and monitoring TFJob jobs.

3. **Machine Learning with TensorFlow in Vertex AI**: This project involves deploying a Vertex AI Workbench instance, creating minimal training and validation data, creating the input data pipeline, creating a TensorFlow model, deploying the model to Vertex AI, deploying the Explainable AI model to Vertex AI, and making predictions from the model endpoint.

---

## Course 3: Deploying Machine Learning Models in Production

### Key Concepts

- Understanding autoscaling model deployments with TensorFlow Serving and Kubernetes
- Implementing Canary Releases of TensorFlow Model Deployments with Kubernetes and Anthos Service Mesh
- Utilizing TFX on Google Cloud Vertex Pipelines

### Projects

1. **Autoscaling TensorFlow model deployments with TF Serving and Kubernetes**: Using TensorFlow Serving and Google Cloud Kubernetes Engine (GKE) to configure a high-performance, autoscalable serving system for TensorFlow models.

2. **Implementing Canary Releases of TensorFlow Model Deployments with Kubernetes and Anthos Service Mesh**: Using Anthos Service Mesh on GKE and TensorFlow Serving to create canary deployments of TensorFlow machine learning models.

3. **TFX on Google Cloud Vertex Pipelines**: Automating the development and deployment of a TensorFlow classification model which predicts the species of penguins. This includes creating a TFX Pipeline using TFX APIs, defining a pipeline runner, and deploying and monitoring a TFX pipeline on Vertex Pipelines.

4. **Data Loss Prevention: Qwik Start - JSON**: Using the Data Loss Prevention API to inspect a string of data for sensitive information, and redacting any sensitive information that was found.
