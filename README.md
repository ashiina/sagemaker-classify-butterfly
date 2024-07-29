# SageMaker Butterfly Classifier

This is a script that trains and deploys an image classification model in Amazon SageMaker.

## Dataset

I used the public dataset [Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification) on Kaggle for training.

## Classifier

The model is SageMaker’s Image Classification framework (based on MXNext), with the following hyperparameters:

```latex
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
```

The hyperparameters are chosen arbitarily as common practice and also relatively small numbers, since this is a demonstration of the deployment more than accuracy. The number of classes and samples in the dataset are dynamically calculated. 

## Setup Instructions

1. Download the dataset from Kaggle and unzip it in the repo's folder
1. Create a virtual environment 
```
source venv.sh
```
1. Install dependencies
```
pip install -r requirements.txt
```




