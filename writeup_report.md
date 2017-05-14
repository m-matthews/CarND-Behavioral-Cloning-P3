#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image2]: ./examples/center_2017_04_09_14_44_02_952.jpg "Centre Driving"
[image3]: ./examples/center_2017_04_09_14_58_40_484.jpg "Recovery Image"
[image4]: ./examples/center_2017_04_09_14_58_40_626.jpg "Recovery Image"
[image5]: ./examples/center_2017_04_09_14_58_40_765.jpg "Recovery Image"
[image6]: ./examples/center_2017_04_09_14_58_40_902.jpg "Recovery Image"
[image7]: ./examples/center_2017_04_09_14_44_02_952.jpg "Normal Image"
[image8]: ./examples/center_2017_04_09_14_44_02_952_flipped.jpg "Flipped Image"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode *(unchanged from original github code)*
* `model.h5` containing a trained convolution neural network
* `video.mp4` video of the vehicle completing the track in autonomous mode
* `writeup_report.md` this file summarizing the results
* `examples/*` images relating to this file
* `model.pdf` file generated with training and validation loss information on various models and parameters

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model(s) consist of convolution neural networks with 5x5 and 3x3 filter sizes and depths between 24 and 64 (`model.py` lines 104-153).

The models includes RELU layers to introduce nonlinearity (`model.py` lines 104-153), and the data is normalized in the model using a Keras lambda layer (`model.py` line 102).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (`model.py` lines 104-153).

The model was trained and validated on different data sets to ensure that the model was not overfitting (`model.py` lines 160-164). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer (`model.py` line 157).  While manually training the learning rate wasn't necessary, I experimented with values of 0.001 and 0.0001 (the default for Keras) to see if there would be any significant changes.

The iterative approach taken to model selection also varied the Batch Size and Dropout rates to investigate possible solutions.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the lap in the opposite direction and additional training data on specialised areas such as the bridge and areas with dirt access roads.

For further details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the `nvidia` model as a starting point.

The use of a Python generator was considered, however the PC used for training was able to hold the required data in memory so the process was more efficient without using a Python generator in this situation.

My first step was to use a convolution neural network model similar to the `nvidia` model.  I considered this model appropriate as a starting point because it is a known model used for self-driving vehicles and behavioural cloning.

The initial image manipulation prior to all models is to:

1) Crop the image to remove image information from above the horizon (70 pixels) and also from the lower area (25 pixels) which would include the front of the car.

2) Normalise the image.

```python
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1))
```

After this, there are four model options included in the code for testing which are visible in the `model.py` code (lines 104-153):

1) 'LeNet' model.  This was used purely for early experimentation and is not used.

2) 'Nvidia' model.  This base model is tested along with the other models for comparison.

3) 'Custom1' model.  This model is based on the Nvidia model, and includes dropout in the Convolution layers.

4) 'Custom2' model.  This model is based on the Nvidia model, and includes dropout in both the Convolution layers and also the Fully Connected layers.

The two 'Custom' models are designed to try to prevent overfitting of the model by using Dropout techniques.  The Dropout rates of 0.1 and 0.2 were tested in the final iterations (0.4 was found to be too high in early tests).

The code iterates over parameters to display options to determine the best model.  The code in `model.py` includes the following which produces different combinations which are then displayed in [model.pdf](examples/model.pdf).

```python
    for LEARNING_RATE in [0.001, 0.0001]:
        for BATCH_SIZE in [64, 128, 256, 512]:
            for NETWORK, DROPOUT in [["nvidia", None], ["custom1", 0.1],
                                     ["custom1", 0.2], ["custom2", 0.1],
                                     ["custom2", 0.2]]:
```

The output in [model.pdf](examples/model.pdf) can be used to guage how well the models are working.  Some clearly show overfitting (for example page 27 `Model:nvidia, Batch Size:128, Learn Rate:0.0001`), however others are suitable for use.

The final option selected was then used to re-generate the model with the following parameters:

```python
    NETWORK = "custom2"
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 512
    DROPOUT = 0.2
```

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road which can be seen in `video.mp4`.

Note that in early attempts, there were a few spots where the vehicle left the track.  The most significant problem originally was the incorrect BGR and RGB colour spaces from OpenCV and the simulator.  I also provided additional training data for any interesting areas, as discussed in section 3 below.

####2. Final Model Architecture

The final model architecture (`model.py` lines 104-153) consisted of a convolution neural network with the following layers and layer sizes.

| Layer | Type          | Details                                             | Input    | Output   |
| ----- | ------------- | --------------------------------------------------- | -------- | -------- |
| 1     | Convolutional | 24 Layers, Kernal 5x5, Strides 2x2, Relu activation | 66x200x3 | 31x98x24 |
| 2     | Convolutional | 36 Layers, Kernal 5x5, Strides 2x2, Relu activation | 31x98x24 | 14x47x36 |
| 3     | Dropout       | Dropout Layer (20%)                                 | 14x47x36 | 14x47x36 |
| 4     | Convolutional | 48 Layers, Kernal 5x5, Strides 2x2, Relu activation | 14x47x36 | 5x22x48  |
| 5     | Dropout       | Dropout Layer (20%)                                 | 5x22x48  | 5x22x48  |
| 6     | Convolutional | 64 Layers, Kernal 3x3, Relu activation              | 5x22x48  | 3x20x64  |
| 7     | Dropout       | Dropout Layer (20%)                                 | 3x20x64  | 3x20x64  |
| 8     | Convolutional | 64 Layers, Kernal 3x3, Relu activation              | 3x20x64  | 1x18x64  |
| 9     | Flatten       | Flatten Layer                                       | 1x18x64  | 1152     |
| 10    | Dense         | 100 Neurons                                         | 1152     | 100      |
| 11    | Dropout       | Dropout Layer (20%)                                 | 100      | 100      |
| 12    | Dense         | 50 Neurons                                          | 100      | 50       |
| 13    | Dropout       | Dropout Layer (20%)                                 | 50       | 50       |
| 14    | Dense         | 10 Neurons                                          | 50       | 10       |
| 15    | Dense         | 1 Neuron                                            | 10       | 1        |

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

| Centre Lane Driving |
| ------------------- |
| ![alt text][image2] |

I then recorded a full lap of the vehicle driving in the opposite direction to help generalise the model.  The existing laps all have a left-turn bias so driving in the opposite direction should assist in that generalisation.

I then recorded a full lap of the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from out-of-normal conditions.  These images show what a recovery looks like starting from the left border and moving towards the centre:

| Recovery Driving 1  | Recovery Driving  2 |
| ------------------- | ------------------- |
| ![alt text][image3] | ![alt text][image4] |

| Recovery Driving 3  | Recovery Driving  4 |
| ------------------- | ------------------- |
| ![alt text][image5] | ![alt text][image6] |

A final set of additional driving was performed at areas that were different in appearance to the majority of the driving surface.  This included the bridge, and also the areas where the side of the road was dirt rather than marked.

The log generated from executing `model.py` includes the following details showing the folders used.  Having the folders separated allows easy exclusion of data from model training if required:

```
    Reading CSV file: /home/michael/CarND-Behavioral-Cloning-P3/track1_centre/driving_log.csv
    Reading CSV file: /home/michael/CarND-Behavioral-Cloning-P3/track1_centre_reversed/driving_log.csv
    Reading CSV file: /home/michael/CarND-Behavioral-Cloning-P3/track1_corrections/driving_log.csv
    Reading CSV file: /home/michael/CarND-Behavioral-Cloning-P3/track1_additional/driving_log.csv
```

At this stage, no training data from Track 2 has been used in the current model training, although that could be included to assist in generalising the model.

To augment the data sat, I also flipped images and angles thinking that this would help to generalise the model as the majority of the driving is to the left given the track is circular.  For example, here is an image that has then been flipped:

| Original Image      | Flipped Image       |
| ------------------- | --------------------|
| ![alt text][image7] | ![alt text][image8] |

The images at this point were biased towards centre line driving and there were a majority of points with low steering angles.  I removed approximately 75% of these points with the following code in the `create_data()` function:

```python
    if abs(measurement) < 0.075:
        if random.random()>0.25:
            continue
```

The final distribution of steering angles can be seen on the first page of the [model.pdf](examples/model.pdf).

After the collection and  process, I had 13,424 number of data points. I then preprocessed this data by converting the image from the `openCV` `BGR` image format to `RGB`.  This ensures that the images passed in from `drive.py` are in the same format for the prediction to function correctly.  Without this, the vehicle has a tendency to take a swim in the river after the first bend due to the swapped blue and red channels in the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 7 and 10 as evidenced by the final page of [model.pdf](examples/model.pdf), noting that this differed slightly with re-runs due to random shuffling of the data.

I used an adam optimizer.  While manually training the learning rate wasn't necessary, I experimented with values of 0.001 and 0.0001 (the default for Keras) to see if there would be any significant changes.
