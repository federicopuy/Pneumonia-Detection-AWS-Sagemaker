## Pneumonia Detection by Fine-tuning VGG-16 using AWS Sagemaker
In this project we analyze a set of X-Ray images of patients suffering from pneumonia and we fine-tune an image recognition model to identify whether a new image belongs to a patient suffering from the disease. 

Tools used: AWS Sagemaker, AWS S3, PyTorch, Pandas

### Dataset
There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

![jZqpV51](https://github.com/federicopuy/Pneumonia-Detection-AWS-Sagemaker/assets/12384264/c4739a83-34e9-4752-98ad-d00756ddd82c)


### Project Set Up Instructions
Import the project into a Jupyter notebook in Sagemaker. You will need to run the cells inside `train_and_deploy.ipynb`. Note that there are also 3 python scripts which define how to perform hyperparameter tuning, training and inference.

Once finished, remember to shutdown the Studio instances and delete the data from the S3 bucket.

### Hyperparameter Tuning
We perform hyperparameter tuning to systematically search for the optimal set of hyperparameters that result in the best performance of this model. The model is located in `hpo.py` and will be executed as a training job using AWS Estimator.

<img width="1194" alt="Screenshot 2024-06-23 at 7 42 24 PM" src="https://github.com/federicopuy/Pneumonia-Detection-AWS-Sagemaker/assets/12384264/5a57dd72-ef9c-45bc-953b-9081a6a05f29">


### Training with Model Profiling and Debugging
Using the best hyperparameters we create and finetune a new model, this time adding model profiling and debugging. The new model is located in `train_model.py`

We then generate the Sagemaker Debugger Profiling report including the details of the training job. For this scenario, no significant recommendations were made by the report to improve training.

### Model Deployment
Finally, we deploy the model to an AWS Endpoint so that we can easily predict our images. Transformations on the images are handled by the endpoint and its configuration is defined in `inference.py`.

The easiest way to query the endpoint is by using the predictor created in the notebook:

```python
response = predictor.predict(data = image_bytes)
```
<img width="1265" alt="Screenshot 2024-06-23 at 7 50 37 PM" src="https://github.com/federicopuy/Pneumonia-Detection-AWS-Sagemaker/assets/12384264/f9a2ff47-9f63-4e9b-8d5a-b7f0ca5716ae">


