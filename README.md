# Using Efficientnet for Pawpularity Prediction

- ### Requirements
    - pytorch
    - albumentations
    - timm
    - fastai
    - tqdm
- ### Repository Structure
    - model.py includes the class that defines efficienet model; 
    - dataset.py includes the class that defines dataset and load image data; dataset_mixup.py applies mixup augmentation to the dataset; 
    - losses.py includes the class RMSELoss, which calculates the mean squared error between the predicted score and the ground-truth score. This RMSE loss also has a regularization term to prevent overfitting.
    - train_cv.py is the file used for 5-fold cross validation. train_all.py is for training the model on all the training data and produces the final model for generating testing data predictions.
- ### Run
    - python train_cv.py / python train_all.py