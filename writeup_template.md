#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my Project 2.
(https://github.com/myladyace/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many datas marked as each labels in the training set.

![Train Image Count](https://github.com/myladyace/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/Train_data_count.png)

Also 6 random images from training data set are below:

![Train Image Plot](https://github.com/myladyace/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/Train_data_samples.png)

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it will make the training process faster

As a last step, I normalized the image data because it would have 0 mean and equal variance.

I didn't generate additional data because the  validation and test accuracy is good enough in my view. 

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  					|
| Flatten		      	| outputs 400									|
| Fully connected		| outputs 200        							|
| RELU					|												|
| Dropout     			| Keep_prob 0.7        							|
| Fully connected		| outputs 100  									|
| RELU					|												|
| Dropout     			| Keep_prob 0.7        							|
| Fully connected		| outputs 43        							|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adamoptimizer is used with learning rate = 0.002, Batch size = 150 and Epochs = 15 and dropout = 0.7.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 13th and 14th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.944 
* test set accuracy of 0.936

-What was the first architecture that was tried and why was it chosen?
A: I used an architecture similar to the 5-LeNet. I used it because the accuracy proven to be very good.

-What were some problems with the initial architecture?
A: The architecture works well but a little bit overfitting.

-How was the architecture adjusted and why was it adjusted?
A: Bring in dropout with keeping probability 0.7 

-Which parameters were tuned? How were they adjusted and why?
A: Epoch, learning rate, batch size, and drop out probability were all parameters tuned. For Epoch, when I set it to 15, it seems the validation accuracy began decreaing after the 10th epoch, so finally it was set as 10. Increase the batch size from 128 to 150 would accelerate the training process. The learning rate of 0.001 works pretty well so I did not change it. The keep probablity works well as 0.5, 0.6 and 0.7. I think 0.7 is low enough to prevent the overfitting. 

-What are some of the important design choices and why were they chosen? 
Actually I found the number of data in each class is not the even, so I gonna try to add more data using rotating or other means for those classes with less data. I thinks this will improve the performance of my model.

If a well known architecture was chosen:
* What architecture was chosen? 
A: Lenet but using the dropout
* Why did you believe it would be relevant to the traffic sign application? 
A: It gives pretty good accuracy
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
A: Test set accuracy more than 0.93
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

New images are from the http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset. I downloaded the test dataset which include 12,630 test images. I first chose the first 10 images with the shape of (32,32,3) which can perfectly fit my model. After visulization these 10 pictures, I picked out 5 of them(with different labels that already been included in the 43 classes).

Here are five German traffic signs that I found on the web:

![5 Test Images](https://github.com/myladyace/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/5_Test_images.png)

The second image might be difficult to classify because the resolution is not good and there is another sign on top of the roundabout sign.
The third and fourth may be difficult given the low resolution so that the number may not be correctly classfied.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only     		| Ahead only   									| 
| Roundabout mandatory  | Priority road 								|
| 50 km/h				| 50 km/h										|
| 30 km/h	      		| 30 km/h						 				|
| Priority road 		| Priority road     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is worse compared to to the accuracy on the test set of 93.6%. 

The only failure happened on the roundabout sign, i think it is because there is also a priority-road-like sign on top of the roundabout sign, so it is classified as the roundabout sign. Also the resolution is not good.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th~21st cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a ahead only sign (probability of 99.8%), and the image does contain a ahead only sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Ahead only   									| 
| .98     				| Priority road 								|
| .99					| 50 km/h										|
| .62	      			| 30 km/h						 				|
| 1.				    | Priority road      							|

For the forth image, the model is not quite sure as others. The image is corretly classified but only with 62% possibility while more than 37% goes to 70 km/h.

The second image is wrongly classfied with almost 100% certainty, the reason I have mentioned above.

