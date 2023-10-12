import time
import sagemaker
import boto3
from sagemaker.huggingface import get_huggingface_llm_image_uri
import json
from sagemaker.huggingface import HuggingFaceModel

sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket = None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

iam = boto3.client('iam')
role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

# retrieve the llm image uri
llm_image = get_huggingface_llm_image_uri(
    "huggingface",
    version="1.0.3"
)

# print ecr image uri
print(f"llm image uri: {llm_image}")

# sagemaker config
instance_type = "ml.g5.xlarge"
number_of_gpu = 1
health_check_timeout = 300

# Define Model and Endpoint configuration parameter
config = {
    # path to where sagemaker stores the model
    'HF_MODEL_ID': "meta-llama/Llama-2-7b-hf",
    'SM_NUM_GPUS': json.dumps(number_of_gpu),  # Number of GPU used per replica
    'MAX_INPUT_LENGTH': json.dumps(1024),  # Max length of input text
    # Max length of the generation (including input text)
    'MAX_TOTAL_TOKENS': json.dumps(2048),
    # 'HF_MODEL_QUANTIZE': "bitsandbytes",# Comment in to quantize
}

# create HuggingFaceModel with the image uri
llm_model = HuggingFaceModel(
    role=role,
    image_uri=llm_image,
    env=config
)

print("HF Model created, deploying it now...")

# Deploy model to an endpoint
# https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy
llm = llm_model.deploy(
    initial_instance_count=1,
    endpoint_name="Llama-2-7b-chat-hf-not-fine-tuned",
    instance_type=instance_type,
    # volume_size=400, # If using an instance with local SSD storage, volume_size must be None, e.g. p4 but not p3
    # 10 minutes to be able to load the model
    container_startup_health_check_timeout=health_check_timeout,
)
