#**Self-Driving Car Engineer Nanodegree**

##**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/network_architecture.png "Model Visualization"
[image2]: ./images/center.jpg "Center Image"
[image3]: ./images/recovery.png "Recovery Image"
[image4]: ./images/affine_transformation.png "Image preprocessing"
[image5]: ./images/track1.gif "Track1"
[image6]: ./images/track2.gif "Track2"

---
###Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network model weights
* helpers.py containing helper functions for train / validation generator, data augmentation
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing:
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

###Model Architecture and Training Strategy

My model is inspired by Nvidia - End to End Learning for Self-Driving Cars paper.

The model consists of 5 convolution neural network layers, 3 layers with 5x5 filter sizes, 2x2 stride and depths between 24 and 48 (model.py lines 21-31) and 2 layers with 3x3 filter sizes, 1x1 stride and 64 depth (model.py lines 33-39).

The model includes RELU layers to introduce nonlinearity followed by MaxPooling and Dropout layers and the data is normalized in the model using a Keras lambda layer (code line 19).

Convolutional layers are followed by fully connected layers leading to an output control value which is the steering angle (model.py lines 41-52).

Dropout layers was added after each conv layer in order to reduce overfitting.

Training and validation was made on different data sets to ensure that the model was not overfitting (code line 61-67). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Adam optimizer was used, so the learning rate was not tuned manually (model.py line 56).


Model architecture visualization:

![alt text][image1]

---

###Training data and training process

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to use the proper steering angle to always center the car. These images show what a recovery looks like starting from left to center and right side of the lane:

![alt text][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles, changed brightness and affine transformation thinking that this would make the model more robust and ready to handle tracks with different light conditions, position of car on the road or road angles.

![alt text][image4]

Beside train data set a different set (provided by Udacity as example) was used as validation set.
The validation set helped determine if the model was over or under fitting. After some test and try process I decided to use 5 epochs for training the model, after this I saw that there are not big improvements. I used an adam optimizer so that manually training the learning rate wasn't necessary.

---

###Testing

The model gives good results on track 1 (simulator: 640x480, Fastest):

![alt text][image5]

And also on track 2 (simulator: 640x480, Fastest):

![alt text][image6]
