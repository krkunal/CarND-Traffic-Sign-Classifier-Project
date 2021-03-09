**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/exploratory_viz.png "Visualization"
[image2]: ./output_images/train_class_distribution.png "EDA Class Distribution"
[image3]: ./output_images/model_train_history.png "Training History"
[image4]: ./output_images/web_images_prediction.png "Web Images Predictions"
[image5]: ./output_images/top5_predictions.png "Top 5 Predictions"


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

This image shows a random sample of image from each of train, validation, and test sets.

![alt text][image1]

This image shows the class distribution the train set. Some classes are less represented as compared to others, so using image augmentation to generate additional training data for such class will help in improving the overall model performance

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For preprocessing, the only step I took is normalization of the images so that the gradients don't become too high and model converges faster. The image data was normalized in the range [-1, 1].


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Initially I started with the LeNet architecture. Later, to improve model performance, I added one additional Convolution layer and two Dropout layers. I've used TensorFlow 2.3.2 runtime. The model summary is:

| Layer (type)	                      |    Output Shape           |  Param #			| 
|:-----------------------------------:|:-------------------------:|:-------------------:| 
| input_2 (InputLayer)                |   [(None, 32, 32, 3)]     |  0                  |
| conv2d_3 (Conv2D)                   |   (None, 28, 28, 6)       |  456                |
| max_pooling2d_2 (MaxPooling2D)      |   (None, 14, 14, 6)       |  0                  |
| conv2d_4 (Conv2D)                   |   (None, 10, 10, 16)      |  2416               |
| dropout_2 (Dropout)                 |   (None, 10, 10, 16)      |  0                  |
| conv2d_5 (Conv2D)                   |   (None, 8, 8, 24)        |  3480               |
| dropout_3 (Dropout)                 |   (None, 8, 8, 24)        |  0                  |
| max_pooling2d_3 (MaxPooling2D)      |   (None, 4, 4, 24)        |  0                  |
| flatten_1 (Flatten)                 |   (None, 384)             |  0                  |
| dense_3 (Dense)                     |   (None, 120)             |  46200              |
| dense_4 (Dense)                     |   (None, 84)              |  10164              |     
| dense_5 (Dense)                     |   (None, 43)              |  3655               |


* Total params: 66,371
* Trainable params: 66,371
* Non-trainable params: 0

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used `Adam Optimizer` for model training and used the following hyperparameters - 
* Learning rate = 0.0006
* Epochs = 12
* Batch size = 64

Since I'd passed the labels as Class Ids and not as one hot encoded vectors, I used the loss function `SparseCategoricalCrossentropy` & the Validation metric `SparseCategoricalAccuracy`.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Intially, I used the LeNet architecture to train the model and it gave validation accuracy ~90%. However, even after increasing the number of training epochs, the validation accuracy didn't improve, whereas, the training accuracy touched 99%. This meant that the model has started overfitting. Then, to address that I added one Dropout layer with 25% dropout rate. But, this also didn't help improve the validation accuracy despite of the training accuracy coming down; i.e.; some bias had got introduced into the model. Then, to increase the model capacity, I added one addtional Convolution layer with 3x3 kernel, removed one pooling layer, and added one more Dropout layer with 10% dropout rate. This helped improve the model capacity & performance and bring the validation accuracy above 93%. Then, I further fine tuned the model training by reducing the learning rate to 0.0006 and increasing the number of epochs to 12 to get a stable model with nicely improving validation accuracy upto 95%.

My final model results were:
* training set accuracy of 98.75%
* validation set accuracy of 95.24%
* test set accuracy of 94.28%

The training history plot is - 

![alt text][image3]
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs taken from the web and the model predictions for them:

![alt text][image4]

Here, I've tried to take four images that have very different characteristic as compared to the training set. Other five images are somewhat similar to the training image data & are taken from [This Link](https://www.semanticscholar.org/paper/The-German-Traffic-Sign-Recognition-Benchmark%3A-A-Stallkamp-Schlipsing/22fe619996b59c09cb73be40103a123d2e328111/figure/2) 

I expected the model to do relatively poorly on the images that had very different distribution than the training data.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of way    		| Right of way 									| 
| Yield     			| Yield 										|
| Stop					| Stop											|
| General caution  		| General caution				 				|
| Slippery Road			| Slippery Road      							|
| Road work 			| Road work         							|
| Speed limit 60 KM/h	| Speed limit 50 KM/h  							|
| Straight or Left		| Straight or Left  							|
| Roundabout			| Roundabout        							|


The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 94%. The model has done well even on 3 of the 4 images that have very different distribution.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th & 15th cells of the Ipython notebook. The top five predictions for all the nine image is:

![alt text][image5]

For all the images, the model is very confident of the predictions. Even on the wrong prediction, the model is making a very confident mistake, although the actual label has 2nd highest probability. But, the fact that the model is making very confident predictions right or wrong, shows that the model has become very rigid and might not generalize very well on OOD image data. This might have happened because I didn't employ the desired level of data preprocessing / augmentation on the training data. For the first image with label `Right of way`, the model predictions are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .998         			| Right of way 									| 
| .001     				| Beware of ice									|
| 1.5e-5				| Double curve									|
| 1.3e-5      			| Speed limit 100 KM/h			 				|
| 9.8e-6			    | Priority Road      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


