# Finding Lane Lines on the Road

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

## Reflection

## 1. the description of my pipeline and draw_lines() fnction I modified

### 1.1 my pipeline

The source code of the pipeline I implemented can be summarized as below.

```python
def detect_line_pipeline(img, vertices):
    # 1. Convert an RGB image to grayscale so that the algorithm can calculate the difference between pixels with ease.
    grayed_img = grayscale(img)

    # 2. Remove noise in photos before calculating the difference between pixels
    blured_img = gaussian_blur(grayed_img, kernel_size=9)

    # 3. Calculates the difference between pixels and detects edges by setting two levels of thresholds
    edge_img = canny(blured_img, low_threshold=50, high_threshold=150)
    
    # 4. Assuming that the camera is installed in front of the car body, narrow down the area of ​​the edge to consider
    masked_img = region_of_interest(edge_img, vertices)

    # 5. Convert the coordinate system of the image displaying the detected edge to the Hough coordinate system, and find the start and end points of the lane line
    line_mask = hough_lines(masked_img, 1, 1, 5, 10, 2)

    # 6. Obtain an image in which the original photo and the mask image that can detect the lane are superimposed
    result = weighted_img(img, line_mask)
    return result
```

Each parameter was adjusted to detect a clean lane.

### 1.2 Improved draw_lines() function

The source code of the draw_lines() function I implemented can be summarized as below.

```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    
    # 1. As noted in the comments in the function, we first calculated the slope of the detected lane.
    slopes = (lines[:, :, 3] - lines[:, :, 1]) / (lines[:, :, 2] - lines[:, :, 0])
    
    # 2. Calculated whether the slope of each lane is a positive value or a negative value, and created mask data to extract only each data
    is_right = slopes > 0
    is_left = np.copy(- is_right)
    
    # 3. Draw lane line according to positive / negative slope
    for is_selected in [is_right, is_left]:
        
        # 4. Convert the xy-axis coordinate data included in the extracted lane start point and end point into one-dimensional data, respectively
        x = np.append(lines[is_selected][:, 0], lines[is_selected][:, 2])
        y = np.append(lines[is_selected][:, 1], lines[is_selected][:, 3])
        
        # 5. Do not calculate if lane is not detected
        if len(x) == 0 or len(y) == 0:
            continue
        
        # 6. Calculate slope and intercept by fitting a straight line to xy coordinate point cloud data
        m, b = np.polyfit(x, y, 1)
    
        # 7. Find the corresponding x coordinate from the slope and intercept based on the range of the y axis to be drawn
        y_max      = img.shape[0]     # 540
        y_min      = int((y_max / 2) + 50) # 320
        x_to_y_max = int((y_max - b) / m)
        x_to_y_min = int((y_min - b) / m)
        
        # 8. Connect the start and end points of the calculated xy coordinate system with a straight line
        cv2.line(img, (x_to_y_max, y_max), (x_to_y_min, y_min), color, thickness)
```

As a point of implementation, do not use for loop as much as possible

### 2. Identify potential shortcomings with your current pipeline

The pipeline I have implemented has two weaknesses.

1. The ability to detect lanes depends on the parameters of the algorithm used in the pipeline
   - For example, if you set a low threshold when performing edge detection, many non-lane edge areas will be detected.
2. Since the left and right lanes are classified according to the sign of the slope of the detected straight line, it cannot be detected if the signs of the slopes of the lanes are the same, such as when approaching a curve.


### 3. Suggest possible improvements to my pipeline

One of the improvements is to take advantage of video data when detecting lanes.

The lane slope detected in each frame of the video is not expected to change drastically.

Therefore, save the statistical information of the detected lane inclination for n frames. If the newly detected lane slope is significantly different from the past information, it can be determined that the result is abnormal.