# Introduction

This repository contains code and resources for implementing MLOps (Machine Learning Operations) using Amazon SageMaker. The LLaMA 2 Model from Meta will be used as the reference Large Language Model in this repository

MLOps is a set of best practices and tools for managing the lifecycle of machine learning models, from development to deployment and beyond. With this repository, you can learn how to build, train, deploy, and monitor machine learning models using SageMaker, and how to integrate these models into your production workflows. 

Whether you're a data scientist, a machine learning engineer, or a software developer, this repository aims to provide a comprehensive guide to MLOps with SageMaker.

# Pre-requisites

* This repository is using Python as a language. So please make sure to install the latest version of Python: ```3.11.4``` at the time of writing

* Install AWS Sagemaker python SDK through ```pip install sagemaker --upgrade```

* A valid AWS Account setup with proper SageMaker Role called `sagemaker_execution_role` created: Follow [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)

* [HuggingFace](https://huggingface.co/join) Account and [User Access Token](https://huggingface.co/settings/tokens) 

* LLaMA 2 Access: Before you can start training, you need to accept the LLaMA 2 license to be able to use it by clicking on the Agree and access repository button on the model page at [LLaMA 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) 

# Structure
The repository is split between the main phases of MLOps:

1. ```deployment``` folder: Model Deployment to be able to perform Inference
2. ```fine-tuning``` folder: Model Fine Tuning
3. ```prompting``` folder: Prompting deployed models


