import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer
import json
import time
import sys

# Constants
BUCKET_NAME = "ml-classify-butterfly"
S3_DATASET_TRAIN = "s3://ml-classify-butterfly/dataset/train/train"
S3_DATASET_VAL = "s3://ml-classify-butterfly/dataset/train/val"
S3_LST_TRAIN = "s3://ml-classify-butterfly/dataset/train/train.lst"
S3_LST_VAL = "s3://ml-classify-butterfly/dataset/train/validation.lst"
MODEL_NAME = "classify-butterfly"
PREFIX = "dataset/train/train/"
REGION = "us-east-1"

# Create a boto3 session using the specified profile and region
session = boto3.Session(region_name=REGION)

# Use the session to create a SageMaker session
sagemaker_session = sagemaker.Session(boto_session=session)

# Create IAM role
import time
import json

def create_iam_role():
    iam_client = session.client('iam')

    # Define role name with timestamp
    role_name = f"SageMakerRole-{int(time.time())}"
    
    # Create the IAM role
    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    response = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
        Description="Role for SageMaker to access S3 bucket and other AWS services"
    )

    role_arn = response['Role']['Arn']

    # Attach necessary policies to the role
    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    )
    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
    )

    # Sleep to ensure role propagation
    time.sleep(10)

    # Verify the role policies
    attached_policies = iam_client.list_attached_role_policies(RoleName=role_name)['AttachedPolicies']
    policy_arns = [policy['PolicyArn'] for policy in attached_policies]
    required_policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    ]

    for policy in required_policies:
        if policy not in policy_arns:
            raise Exception(f"Required policy {policy} is not attached to the role {role_name}")

    # Verify the trust relationship
    trust_relationship = iam_client.get_role(RoleName=role_name)['Role']['AssumeRolePolicyDocument']
    required_trust_relationship = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }

    if trust_relationship != required_trust_relationship:
        raise Exception(f"The trust relationship for the role {role_name} is not as expected")

    return role_arn


def get_num_classes(bucket_name, prefix):
    s3_client = boto3.client('s3', region_name=REGION)
    paginator = s3_client.get_paginator('list_objects_v2')
    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')

    num_classes = 0
    for response in response_iterator:
        print(response)
        if 'CommonPrefixes' in response:
            num_classes += len(response['CommonPrefixes'])

    return num_classes

def get_num_training_samples(bucket_name, prefix):
    s3_client = boto3.client('s3', region_name=REGION)
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    num_training_samples = 0
    for page in pages:
        if 'Contents' in page:
            num_training_samples += len(page['Contents'])

    return num_training_samples


#----MAIN----

print(f"Using training data on S3: {S3_DATASET_TRAIN} and {S3_DATASET_VAL}")
role_arn = create_iam_role()

# hyperparameters
num_classes = get_num_classes(BUCKET_NAME, PREFIX)
print(f"Number of classes (folders) in the training data: {num_classes}")
num_training_samples = get_num_training_samples(BUCKET_NAME, PREFIX)
print(f"Number of training samples: {num_training_samples}")

# Define the image URI for the algorithm container
algorithm_image_uri = sagemaker.image_uris.retrieve(framework='image-classification', region=REGION)

# Create and configure the Estimator
print("Creating and configuring the Estimator...")
estimator = Estimator(
    image_uri=algorithm_image_uri,
    role=role_arn,
    instance_count=1,
    instance_type='ml.p2.xlarge',
    volume_size=50,
    max_run=3600,
    input_mode='File',
    output_path=f"s3://{BUCKET_NAME}/{MODEL_NAME}/output",
    sagemaker_session=sagemaker_session
)

# Set hyperparameters
estimator.set_hyperparameters(
    num_layers=18,
    use_pretrained_model=1,
    num_classes=num_classes,
    mini_batch_size=32,
    epochs=10,
    learning_rate=0.001,
    precision_dtype='float32',
    num_training_samples=num_training_samples
)

# Prepare the dataset for training
train_data = TrainingInput(s3_data=S3_DATASET_TRAIN, content_type="application/x-image", input_mode='File')
validation_data = TrainingInput(s3_data=S3_DATASET_VAL, content_type="application/x-image", input_mode='File')
train_lst = TrainingInput(s3_data=S3_LST_TRAIN, content_type="application/x-image", input_mode='File')
validation_lst = TrainingInput(s3_data=S3_LST_VAL, content_type="application/x-image", input_mode='File')
data_channels = {
    'train': train_data,
    'validation': validation_data,
    'train_lst': train_lst,
    'validation_lst': validation_lst
}

# Start the training job
print("Starting the training job...")
estimator.fit(inputs=data_channels)
print("Training job completed.")

# Deploy the model
print("Deploying the model...")
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=MODEL_NAME,
    serializer=IdentitySerializer("image/jpeg"),
    deserializer=JSONDeserializer()
)

# Get the endpoint URL
endpoint_name = predictor.endpoint_name
endpoint_url = f"https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{endpoint_name}/invocations"

print(f"Model deployed. Endpoint name: {endpoint_name}")
print(f"Endpoint URL: {endpoint_url}")

