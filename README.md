# Brain Tumor Detection using CNN 

## Dataset
The dataset used for this project is the [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection). It consists of MRI scans categorized into two classes:
* `NO` - no tumor (encoded as `0`)
* `YES` - tumor (encoded as `1`)

And the pretrained model used here is [VGG16 MODEL](https://www.kaggle.com/datasets/gaborfodor/keras-pretrained-models)

## Setting up the Environment
Ensure the required packages are installed by running the provided code in the "Setting up the Environment" section. The code sets up the necessary libraries for data preprocessing and model building.

## Data Import and Preprocessing
Load and preprocess MRI images for the machine learning model. The code organizes the data into training, validation, and test sets. It also crops the images to focus on the brain region.

## CNN Model
Build a Convolutional Neural Network (CNN) for brain tumor classification. The code uses a pre-trained VGG16 model for feature extraction and adds a dense layer for binary classification.

### Data Augmentation
Implement data augmentation to enhance the model's generalization. The code demonstrates augmenting images by applying random transformations.

## Model Building
Train the CNN model using the prepared dataset. The code uses transfer learning with a pre-trained VGG16 model and fine-tunes it for brain tumor detection.

## Model Performance
Evaluate the model's performance on the validation set. The code plots accuracy and loss graphs during training. Additionally, it displays the confusion matrix for a detailed performance analysis.

## Conclusions
This project combines CNN classification and Computer Vision to automate brain tumor detection from MRI scans. The model achieves a significant accuracy boost compared to a baseline. Further improvements could come from a larger dataset or fine-tuning hyperparameters.

## Clean Up and Save Model
The code at the end cleans up the workspace and saves the trained model as '2019-06-07_VGG_model.h5'.

*Note: The provided code assumes a certain file structure and may require adjustments based on your environment.*
