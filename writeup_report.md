##**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undist.png "Undistorted"
[image2]: ./output_images/undist2.png "Road Undistorted"
[image3]: ./output_images/binary_combo.png "Binary Example"
[image4]: ./output_images/warped.png  "Warp Example"
[image5]: ./output_images/fit1.png "Fit Visual"
[image6]: ./output_images/fit2.png "Fit Visual2"
[image7]: ./output_images/meas.png "measure"
[image8]: ./output_images/draw.png "draw"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

## Working Files Description

I have two main working files for this project:
* ./Proj.ipynb: My development of pipeline was mostly done in this file,  where I preserved intermediate process steps for further tuning.
* ./process.py: After I completed my tuning, I wrapped the pipeline code in this file so I can easily call it wherever I like.

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cells under **Camera Calibration** of the IPython notebook located in "./Proj.ipynb" and also at lines 17 through 40 under `def calibration(self, path):` in `process.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image (It's a flat chaseboard not a 3D one!).  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

After the camera calibration, we can run `cv2.undistort()` with the parameters obtained, `mtx, dist`. The example is shown as the image below:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in the code cells under **Gradient and Color** of the IPython notebook located in "./Proj.ipynb" and also at lines 100 through 120 under `def binary_image(self...):` in `process.py`.


I used a combination of color and gradient thresholds to generate a binary image. Here's an example of my output for this step.

![alt text][image3]

In the binary process, I computed x gradients on the L channel(HLS) and set the threshold of 35 to 150, and for the color method I used the threshold of 130 to 255 on the S Channel.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained in the code cells under **Perspective Transform** of the IPython notebook located in "./Proj.ipynb" and also at lines 42 through 52 under `def perspective_matrix(self):` in `process.py`.

I began from detecting the vertexes from the binary images and then finetuned the points.

```python
down = 690
up = 455
s1 = [np.where(imgs_combined[idx][down-1,:] == 1)[0][0]+1,down]
s4 = [np.where(imgs_combined[idx][down-1,:] == 1)[0][-1]+1,down]
s2 = [np.where(imgs_combined[idx][up-1,550:720] == 1)[0][0]+551,up]
s3 = [np.where(imgs_combined[idx][up-1,550:720] == 1)[0][-1]+548,up]
s1 = [207,720]
s2 = [586,455]
s3 = [698,455]
s4 = [1127,720]
side_offset = 160
src = np.float32([s1,s2,s3,s4])
dst = np.float32([[src[0,0]+side_offset,720],\
[src[0,0]+side_offset,0],[src[3,0]-side_offset,0],\
[src[3,0]-side_offset,720]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 207, 720      | 367, 720        |
| 586, 455      | 367, 0      |
| 698, 455     | 967, 0      |
| 1127, 720      | 967, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is contained in the code cells under **Perspective Transform** of the IPython notebook located in "./Proj.ipynb" and also at lines 42 through 52 under `def perspective_matrix(self):` in `process.py`.

For this part, I used the provided code based on sliding window and histogram peak method to search for the lines. For the consequential frames, I also used the suggested method for searching the lines based on +/-margins of the fitted lines in the previous frame. Once the line points are marked, `np.polyfit(lefty, leftx, 2)` can be used to find the coefficients of the second order polynomial. The result images are shown as:

![alt text][image5]
![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
The code for this step is contained in the code cells under **Measuring Curvature** of the IPython notebook located in "./Proj.ipynb" and also at lines 319 through 348 under `def measurement(self):` in `process.py`.

First, in order to convert pixels to meters, I manually measured the land width and the line length (in pixels) as shown in the image below:

![alt text][image7]

The width and length are 623 and 70 pixels, respectively. Then, I computed the equivalent meter/pixel based on the real width of 3.7m  and length of 3m. For example the below image shows:
![alt text][image6]
```python
Curvature: left 966 m, right 759 m
Vehicle is -0.27 m left of center
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is contained in the code cells under **Drawing and Text** of the IPython notebook located in "./Proj.ipynb" and also at lines 350 through 399 under `def draw(self, img):` in `process.py`.
 Here is an example of my result on a test image:
![alt text][image8]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. First, there are some techniques I would like to explain in details:

* Measurement:
I developed a set of extra signals to be measured by the measurement function (at line 319 to 348 `def measurement(self):`): (1) curvature; (2) position of the line; (3) Variance of the horizontal difference of the lines for checking whether they are parallel or not; (4) Mean of the horizontal difference of the lines to see the width of the line; (5) log ratio of the curvatures of the two line = `np.abs(np.log10(left_curvature/right_curvature))`, the main reason of using log is because the curvature becomes extremely large whenever the lines are almost straight.

* Validation of the current frame:
The code for this step is contained in the code at lines 442 through 472 under `update(self, obj):` in `process.py`. I first declared left and right line objects (declared at line 54 to 85 `class Line():`) and passed them to the pipeline (line 475 to 514 `def process_image(self, img):`). When the pipeline is running, the line objects will get updated from the measurement block. I then tested the pipeline with the video to see what defines a good frame or what happens when a bad frame occur. After a bunch of tests, I then choose the difference in fit coefficients between last and new fits to decide whether this frame is valid or not. Particularly, I normalized the fit coefficients `obj.diff_fit = np.abs(obj.current_fit - obj.previous_fit)*[8000,2.5,1/80]` so that the thresholds are all 1s. This makes it very easy to visualize in the rendered files.

* When a bad frame happens:
Once a bad frame is detected, I will save the fit coefficients in `obj.previous_fit` but not in `obj.recent_fit`. Then the averaging operation of the fit coefficients does not consider bad frames. Also, `obj.detected` will be flagged as False. The polygon drawing then uses only the average values from previous 8 good frames. Besides, the next frame will use the fit coefficients obtained from the last good frame obtained to start the search. With this algorithm, the influences of bad frames should not appear.   

* Prevention of deadlock:
I have also developed some rules for preventing deadlock situations. For example, whenever a bad frame occurs, the fit coefficients are still preserved but not included for averaging operation. Then, in the next frame, the differential coefficients as compared to the previous bad frame will be checked with a harder threshold. By this mean, it can prevent the pipeline from deadlocking the prediction when land lines are actually changing pretty fast; that is, if the consequential coefficients are very similar it is very possible that the line shapes are actually changing a lot.

Moreover, if this unlocking rule does not work, then a second way is to reset and restart to search new sliding windows. After detecting 10 successive bad lines, the pipeline resets and restarts sliding window searching.

* Possible issues and future improvements:
I tested my pipeline on the challenge videos and found that the results are not very satisfactory. I have thought about a few possible improvements: (1) retuning of color threshold; (2) Applying directional gradient filters as lines tend to have particular orientations; (3) Enforcing more validation rules to find out the bad frames; (4) Resetting left/right line windows independently so that they can compliment each others whenever a bad frame occur (most of the time bad detection only occurs to one side of lines); (5) Application of spatial low-pass filter as average method may not be ideal as it weights all the previous frames equally (they should not have the same values for future prediction).
