from skimage.metrics import structural_similarity
"""
scikit-image is an open-source image processing library for the Python programming language. 
It includes algorithms for segmentation, geometric transformations, color space manipulation, 
analysis, filtering, morphology, feature detection, and more.
"""
#----------------
"""
-->I used "structural_similarity" from metrics module to for performing SSIM
In this project it purpose is to :
> To find out similarity percentage 
> Difference between the inputs
"""
import cv2
"""
OpenCV-Python is a library of Python bindings designed to 
solve computer vision problems.we can develop real-time computer vision applications. It mainly focuses 
on image processing, video capture and analysis including features like face detection and object detection
I will use opencv-python in my program to :
> Resize image
> To convert RGB to GRAY 
> For thresholding
> Finding Contours 
> To draw rectangles on it
"""
import imutils
"""
OpenCV-Python is a library of Python bindings designed to 
solve computer vision problems.we can develop real-time computer vision applications. It mainly focuses 
on image processing, video capture and analysis including features like face detection and object detection
I will use opencv-python in my program to :
> Resize image
> To convert RGB to GRAY 
> For thresholding
> Finding Contours 
> To draw rectangles on it
"""

image1 = cv2.imread("1.jpg")
image2 = cv2.imread("2.jpg")
"""
            reading the Browsed Images 
"""
resizedImage1 = cv2.resize(image1, (500, 500), interpolation=cv2.INTER_AREA)
resizedImage2 = cv2.resize(image2, (500, 500), interpolation=cv2.INTER_AREA)
"""
            Resizing Images for ideal size using cv2.resize()
"""
convertedImage1 = cv2.cvtColor(resizedImage1, cv2.COLOR_BGR2GRAY)
convertedImage2 = cv2.cvtColor(resizedImage2, cv2.COLOR_BGR2GRAY)
"""
            Converting Resized Images from RGB to GRAY
            For many applications of image processing, color information doesn't
            help us identify important edges or other features. 
 """
(similarity, difference) = structural_similarity(convertedImage1, convertedImage2, full=True)  # , full=True
"""
            Applying Structural Similarity model to fetch Similarity and Difference
"""
differed = (difference * 255).astype("uint8")
"""
           Differed Image is being finding here.
"""
threshold = cv2.threshold(differed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
"""
            Finding Threshold of differed image using 
            Applying gaussian blur using inverse binary threshhold on minimum 0 and maximum 255 with
            otsu threshold
            THRESH_OTSU :  Otsu's method avoids having to choose a value and determines it automatically.
                    its arbitary !!
"""
cotourHold = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
"""
            Contours can be explained simply as a curve joining all the continuous points (along the boundary), 
        having same color or intensity. The contours are a useful tool for shape analysis and 
        object detection and recognition.

        > RETR_EXTERNAL : retrieves only the extreme outer contours. cv2.RETR_LIST – retrieves all of the contours method
        >  CHAIN_APPROX_SIMPLE :  it removes all redundant points and compresses the contour, thereby saving memory.
"""
cotour = imutils.grab_contours(cotourHold)
for value in cotour:
    (x, y, w, h) = cv2.boundingRect(value)
    """
                    cv2.boundingRect is the ratio of width to height of bounding rect of the object.
                               it is use to sketch around our Object
    """
    cv2.rectangle(resizedImage1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # (image, start_point, end_point, color, thickness)
    cv2.rectangle(resizedImage2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    """
                    Drawing on captures ..on the finded point using boundingRect 
    """
    heatmap_orginal_img = cv2.applyColorMap(threshold, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_orginal_img, 0.7, resizedImage1, 0.3, 0)
    heatmap_differed_ig = cv2.applyColorMap(differed, cv2.COLORMAP_JET)
    difer = cv2.addWeighted(heatmap_differed_ig, 0.7, resizedImage2, 0.3, 0)
    """
        Showing the images now
    """
cv2.imshow("Original",resizedImage1)
cv2.imshow("Modified", resizedImage2)
cv2.imshow("threshold", threshold)
cv2.imshow("differed", differed)
cv2.imshow("HeatMap of Orginal",fin)
cv2.imshow("HeatMap of differed",difer)
cv2.waitKey(0)