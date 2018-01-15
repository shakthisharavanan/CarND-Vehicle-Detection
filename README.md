**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./car.png
[image2]: ./car1.png
[image3]: ./norm.png
[image4]: ./sliding.png
[image5]: ./sliding1.png
[image6]: ./sliding2.png
[image7]: ./sliding3.png
[image8]: ./sliding5.png
[image9]: ./sliding4.png
[image10]: ./heatmap.png
[video1]: ./final.mp4


---
### Writeup


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 7th code cell of the IPython notebook "vehicle_detection.py". It is based on the class lectures.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some example of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

For this project i used three types of features: HOG (Histogram of Oriented Gradients) which is based on the shape of the object, binned color which represents color and shape features and color histogram features which is based only on color of the object. The code for color histogram and binned color are in the 5th cell of the ipython notebook.

Here is the HOG features of a car image and also a non-car image. 

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

By repeated trial and error, I settled on the HOG as well as the other feature detectors. The parameters may not be ideal, since I had to find the balance between performance and computation time.

Final parameter for feature extraction:

```
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 12 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 2 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

```


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I tried both linear SVM (code cell 51) and Random Forest classifiers(code cell 52). I had better accuracy and results with the random forest clasifier, so I used that as my classifier. Initially I normalised the features using the `StandardScaler()` from sklearn() library. This was the result of normalisation:

![alt text][image3]

Then i split the dataset using `train_test_split` to have 80% as training dataset and 20% as testing dataset. I trained both SVM and Random forest classifiers on the data. I found Random forest classifier to have a slightly better accuracy (99.13%). So i proceeded with that.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window algorithm as taught in the lectures. It is contained in the 55th code cell. Through trial and error I finally ended up with five different configurations with scale between 0.75 to 2.0. These configurations define the region of interest, window size and overlap between the sliding windows.



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using LUV 1st-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

I also flipped the car images to augment the dataset to improve the performance of my classifier.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./final.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

### Here's the test result showing the heatmap, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the image:


![alt text][image10]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Parameter tuning was something that took the most time in this project. I had some issues in drawing boundary boxes for cars that are a little further away from the car on which camera was mounted on. I had to tweak the region of interest carefuly so as to detect cars that are further awat and also not to end up detecting any scenery as it would result in a number of false positives. 

My algorithm may fail in poor lighting conditions. Its also not very robust when 2 cars are next to each other. 

To make my algorithm more robust I could use more images of cars and roads to train the classifier. Also i could use some kind of kalman filtering to predict subsequent positions of nearby vehicles.

