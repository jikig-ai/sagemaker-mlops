# Introduction

This repository contains code and resources for implementing MLOps (Machine Learning Operations) using Amazon SageMaker. 

MLOps is a set of best practices and tools for managing the lifecycle of machine learning models, from development to deployment and beyond. With this repository, you can learn how to build, train, deploy, and monitor machine learning models using SageMaker, and how to integrate these models into your production workflows. 

Whether you're a data scientist, a machine learning engineer, or a software developer, this repository aims to provide a comprehensive guide to MLOps with SageMaker.

# Pre-requisites
This repository is using Python as a language. So please make sure to install the latest version of Python (3.11.4 at the time of writing)

Install AWS Sagemaker python SDK through

```pip install sagemaker --upgrade```

# Structure
The repository is split between the main phases of MLOps:

1. ```deployment``` folder: Model Deployment to be ablke to perform Inference
2. ```fine-tuning``` folder: Model Fine Tuning
3. ```prompting``` folder: Prompting deployed models


