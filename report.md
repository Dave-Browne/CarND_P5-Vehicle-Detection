# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/dataset.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/color_histogram.png
[image4]: ./output_images/windows_heatmap_1.png
[image5]: ./output_images/windows_heatmap_2.png
[image6]: ./output_images/classifier.png
[video1]: ./project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points


## Data Processing
#### `Vehicle_Detection_and_Tracking.ipynb` second cell

All of the non-vehicle data and KITTI vehicle data were used.

The GTI vehicle data consists of multiple images of the same car, taken from slightly different angles/zooms. If similar images appear in the training and test sets, the model will suffer from overfitting. The accuracy results will not be a true reflection of the models ability to correctly classify an unseen example.

To minimise this, every 20th image from the GTI vehicle folder is taken. The similar photos have subsequent names and so most duplicates will be ommitted from the dataset. The vehicle dataset size is reduced from approximately 8900 images to 6100 images. This does not have a large negative impact on the classifiers ability to detect cars.

```
#Importing images
cars = []
notcars = []
n = 20
#Add non car images
images = glob.glob('./non-vehicles/**/*.png', recursive=True)
for image in images:
    notcars.append(image)
#Add car images from KITTI
images_KITTI = glob.glob('./vehicles/KITTI_extracted/*.png')
for image in images_KITTI:
    cars.append(image)
#Add every nth car image from GTI to reduce overfitting in classifier (due to many similar images)
images_GTI = glob.glob('./vehicles/GTI/**/*.png', recursive=True)
for i in range(0, len(images_GTI), n):
    cars.append(images_GTI[i])
```

![alt text][image1]

## Histogram of Oriented Gradients (HOG)
#### `Vehicle_Detection_and_Tracking.ipynb` third cell - *get_hog_features()*

A very good method of extracting features from an image is to take its Histogram of Oriented Gradients using `skimage.feature.hog()`. Experimentation with HOG parameters during the Udacity lessons helped determine good parameter values for this project.

Parameters are chosen in the 4th notebook cell. Increasing the number of `cells_per_block` and `pix_per_cell` would reduce the number of feature vectors. Since the overall video sampling is approximately 1.75it/s, it was decided to leave the parameters as-is.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pix_per_cell=8` and `cells_per_block=2`. The HOG features give the general direction of the pixel in a given block of pixels. The brightness of the line is directly proportional to the strength of the gradients in the image block.

![alt text][image2]

## Color Features
#### `Vehicle_Detection_and_Tracking.ipynb` third cell - *color_hist()* and *bin_spatial()*

More features are extracted from the image using Color Histogram and Spatial Binning methods. The `YCrCb` color space produces a superior model, ie it has a better chance of correctly predicting cars for the given project video.

Below is a graphical representation of the color histogram of the image above:

![alt text][image3]

## Classifier
#### `Vehicle_Detection_and_Tracking.ipynb` seventh cell

A linear SVM and Decision Tree were trained and tested. The linear SVM produced a better result and so was chosen for the project. Using `YCrCb` color space and all the color channels improved the classifier. The trained classifier is saved as `classifier.p`.

`sklearn.preprocessing.StandardScaler` is used to center and scale the features as per the image below.

One of the problems with training the classifier is the amount of memory it uses. If the number of features are too high (eg. if you set n > 32 and spatial_size = (n, n) and hist_bins = n, the classifier crashed on memory error). In an attempt to prevent this the feature vectors are deleted to free up some memory. This however did not help much.

Below is the printout of the details of the classifier training.

![alt text][image6]

## Sliding Window Search
#### `Vehicle_Detection_and_Tracking.ipynb` third cell - *find_cars()*

The sliding window search method requires multiple window sizes. This allows for continuous car detection as they change size. There are two methods in which to implement this:

1) Assign multiple window sizes and run feature extraction on each window. Using multiple windows does not allow for taking the HOG features once per image. This is because if the HOG features are calculated once per image and the window size is larger than that used in the classifier, more features will be extracted than expected and an error will be thrown.

2) Instead of defining multiple window sizes, assign one window size and rather change the image size. This has the same effect of using multiple window sizes, but also allows the calculation of HOG features once per image.

Option 2, using image scaling, was chosen to solve this project. The image under Heatmap shows the original image, the detected windows and the final output.

```
scale_list = np.arange(1.0, 2.1, 0.5)            # scale_list = [1.0, 1.5, 2.0]
```

## Heatmap
#### `Vehicle_Detection_and_Tracking.ipynb` third & fifth cells

The window positions (boxes) which have positive car detections are stored in a list. Every pixel inside each box gets +1 added to it on a blank image. This is called the heatmap. The heatmap is thresholded to eliminate false positives. 

The code below shows how the heatmap is preserved over multiple images. The variable `heat` is continuously appending 'heat' from every new image. The current images heat is added to the list `heat_list`. When the counter reaches `threshold2` the oldest heat is popped from `heat_list` and subtracted from `heat`. This allows `heat` to remember the heat contribution from multiple images.

```
if counter == 1:
    #Define a blank image on which to add 'heat'
    global heat
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
global counter
counter += 1

#Add heat to each box in box_list
heat = add_heat(heat, box_list)
new_heat = np.zeros_like(image[:,:,0]).astype(np.float)
new_heat = add_heat(new_heat, box_list)
heat_list.append(new_heat)

if counter > threshold2:
    counter = threshold2
    neg_heat = heat_list.pop(0)
    heat -= neg_heat

#Apply threshold to help remove false positives
heat_thresh = apply_threshold(heat, threshold)
```

`scipy.ndimage.measurements.label()` is used to identify individual blocks in the heatmap. Each block is assumed to correspond to a vehicle. Bounding boxes are defined to cover the area of each block detected.

```
#Find final boxes from heatmap using label function
structure = np.ones((3,3))
labels = label(heat_thresh, structure=structure)
draw_img = draw_labeled_bboxes(np.copy(image), labels)
```

Below are two examples of the original image, the detected windows, the bounding boxes as a result of `scipy.ndimage.measurements.label()`, the heatmap prior to thresholding and the heatmap after thresholding.

The first image has no cars but detects two false positives. This is caused by the imperfections in the classifier. The heatmap threshold prevents these false positives from passing to the detected cars output.

![alt text][image4]

In the second image, both cars are detected by the sliding windows. There are also no false positives detected, giving increased confidence in the classifier. The heatmap threshold prevents most of these sliding windows from passing through and the detected cars output seems quite poor. This is actually a desired result, as the sliding windows are accumulated in the variable `heat` over multiple (5) images. Thus when watching the video, this image will have alot more accumulated windows over the cars and the result is a positive detection of both cars.

![alt text][image5]

## Video Implementation

The output video works fairly well at detecting cars but suffers from false positives. A better classifier would improve the result.

Here's a [link to the video result](./project_output.mp4)


## Future Improvements

There are two areas of concern in this project, the first being a classifier that detects cars where there are none. This could be improved by using more/better data. A CNN would also improve the performance of the classifier.

The second area to improve would be to combine the 'adjacent' bounding boxes produced by `scipy.ndimage.measurements.label()`.
