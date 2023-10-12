import time
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace
from huggingface_hub import HfFolder

sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket = None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

# try:
#    role = sagemaker.get_execution_role()
# except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

model_id = "meta-llama/Llama-2-7b-hf"  # sharded weights

# define Training Job Name
job_name = f'huggingface-qlora-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'

# hyperparameters, which are passed into the training job
hyperparameters = {
    'model_id': model_id,                             # pre-trained model
    # path where sagemaker will save training dataset
    'dataset_path': '/opt/ml/input/data/training',
    'epochs': 3,                                      # number of training epochs
    'per_device_train_batch_size': 2,                 # batch size for training
    # learning rate used during training
    'lr': 2e-4,
    # huggingface token to access llama 2
    'hf_token': HfFolder.get_token(),
    # wether to merge LoRA into the model (needs more memory)
    'merge_weights': True,
}

# create the Estimator
huggingface_estimator = HuggingFace(
    entry_point='run_clm.py',      # train script
    # directory which includes all the files needed for training
    source_dir='scripts',
    instance_type='ml.g5.4xlarge',   # instances type used for the training job
    instance_count=1,                 # the number of instances used for training
    base_job_name=job_name,          # the name of the training job
    role=role,              # Iam role used in training job to access AWS ressources, e.g. S3
    volume_size=300,               # the size of the EBS volume in GB
    # the transformers version used in the training job
    transformers_version='4.28',
    # the pytorch_version version used in the training job
    pytorch_version='2.0',
    py_version='py310',           # the python version used in the training job
    hyperparameters=hyperparameters,  # the hyperparameters passed to the training job
    # set env variable to cache models in /tmp
    environment={"HUGGINGFACE_HUB_CACHE": "/tmp/.cache"},
)

training_input_path = f's3://{sess.default_bucket()}/processed/llama/dolly/train'

# define a data input dictonary with our uploaded s3 uris
data = {'training': training_input_path}

# starting the train job with our uploaded datasets as input
huggingface_estimator.fit(data, wait=True)
