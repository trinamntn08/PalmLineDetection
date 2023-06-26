from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt

title_window = 'Output'
title_window_preprocessing = 'Preprocessing'

slider_max = 255
slider_max_kernelSize = 21

# Initialize filteredImage as a global variable
handImage = None
filteredImage=None
'''
The color of human skin is created by a combination of blood (red) and melanin (yellow, brown). 
Skin colors lie between these two extreme hues and are somewhat saturated. 
HSV: (Hue-Saturation-Value) defined in a way that is similar to how humans perceive color.
'''
def detectHand(inputImage):
    # define the upper and lower boundaries of the HSV pixel intensities 
    # to be considered 'skin'
    hsvim = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 41, 50], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask= cv2.inRange(hsvim, lower, upper)
    # blur the mask to help remove noise
    skinMask= cv2.blur(skinMask, (2, 2))
    # get threshold image
    ret, thresh = cv2.threshold(skinMask, 50, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Binary Hand", thresh)

    # draw the contours on the empty image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    # Create a mask with the same dimensions as the input image
    mask = np.zeros_like(inputImage)
    print(contours)
    # Draw the contour on the mask as filled white region
    cv2.drawContours(mask, [contours], -1, (255, 255, 255), cv2.FILLED)

    # Apply the mask to the input image using bitwise AND operation
    outputImage = cv2.bitwise_and(inputImage, mask)
    cv2.drawContours(outputImage, [contours], -1, (255, 255, 0), 2)
    #cv2.imshow("Hand Contour", outputImage)
    return outputImage


def on_trackbar_preprocessing(kernelSize):
    global handImage  # Access the global variable
    global filteredImage
    #handImage = cv2.cvtColor(handImage,cv2.COLOR_BGR2GRAY)
    # apply binary thresholding
   # ret, filteredImage = cv2.threshold(grayImage, kernelSize, 255, cv2.THRESH_BINARY)
    kernelSize = kernelSize+1 if kernelSize %2 ==0 else kernelSize 
    handImage = cv2.bilateralFilter(handImage,9,75,75)
    filteredImage = cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)
    output = detectHand(filteredImage)
    cv2.imshow(title_window_preprocessing, output)


def on_trackbar(valLower,valUpper):
    global filteredImage  # Access the global variable
    edges = cv2.Canny(filteredImage ,valLower,valUpper,apertureSize = 3)
    mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)    
    # Apply the mask to the hand image
    outputImage = cv2.addWeighted(mask, 0.7, image, 0.3, 0)
    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the output image
    #outputImage = handImage.copy()
    cv2.drawContours(outputImage, contours, -1, (0, 255, 0), 2)
    cv2.imshow(title_window, np.hstack((filteredImage, outputImage)))


def detectLines(inputImage,outputImage):
    # Perform Hough Line Transform
    lines = cv2.HoughLinesP(inputImage, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    # Filter out long edges
    filtered_lines = []
    threshold_length = 100  # Adjust this threshold as needed

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > threshold_length:
            filtered_lines.append(line)

    # Draw the filtered lines on the original image
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(outputImage, (x1, y1), (x2, y2), (0, 255, 0), 2)

def load_image():
    global image
    image_path = cv2.filedialog.askopenfilename()  # Open file dialog to select an image
    if image_path:
        image = cv2.imread(image_path)
        cv2.imshow(title_window, image)

'''  MAIN LOOP '''
#  Read an image 
image_path = 'D:/dev/projects/PalmLine/input/main_5.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Can not read image")
print('Size original image: ',image.shape[0],'x', image.shape[1])
image = cv2.resize(image, (512, 512))
handImage= detectHand(image)
#cv2.imshow("Input Image",image)
#hist = cv2.calcHist([image],[0],None,[256],[0,256])
#plt.hist(image.ravel(),256,[0,256])
#plt.show()

# Create a window for preprocessing
cv2.namedWindow(title_window_preprocessing)
# Create a trackbar
cv2.createTrackbar('Kernel', title_window_preprocessing, 0, slider_max, on_trackbar_preprocessing)
on_trackbar_preprocessing(0)

# Create a window for result
cv2.namedWindow(title_window)
# Create a trackbar
cv2.createTrackbar('Lower', title_window, 45, slider_max, on_trackbar)
cv2.createTrackbar('Upper', title_window, 95, slider_max, on_trackbar)
# Call on_trackbar initially to show the default edges
on_trackbar(0, 255)

while True:
    kernelSize = cv2.getTrackbarPos('Kernel', title_window_preprocessing)
    # Update the displayed image
    on_trackbar_preprocessing(kernelSize)

    # Get the current trackbar values
    valLower = cv2.getTrackbarPos('Lower', title_window)
    valUpper = cv2.getTrackbarPos('Upper', title_window)

    # Update the displayed image
    on_trackbar(valLower, valUpper)
    

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()



''' FOR VIDEO
# Read image from webcam
video = cv2.VideoCapture(0)
      
# Check if webcam opened successfully
if not video.isOpened():
    print("Failed to open webcam")

# Create a window
cv2.namedWindow(title_window)

# Create a trackbar
cv2.createTrackbar('Lower', title_window, 0, slider_max, on_trackbar)
cv2.createTrackbar('Upper', title_window, 255, slider_max, on_trackbar)


while(True):
    # Capture the video frame # by frame
    ret, image = video.read()
    # Check if frame was successfully read
    if not ret:
        print("Failed to capture frame from webcam")
        break

    # Get the trackbar values
    valLower = cv2.getTrackbarPos('Lower', title_window)
    valUpper = cv2.getTrackbarPos('Upper', title_window)

    # Apply the trackbar values to the frame
    #valLower = valLower / alpha_slider_max
    #valUpper = valUpper / alpha_slider_max

    grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #Equalize histogram
    img= cv2.equalizeHist(grayImage)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img,valLower,valUpper,apertureSize = 3)
    # Resize edges to match the size of the image
    edges = cv2.resize(edges, (image.shape[1], image.shape[0]))
    edges = cv2.bitwise_not(edges)
    # Create a mask from the edges
    mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    lined = np.copy(image) * 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, np.array([]), 50, 20)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lined, (x1, y1), (x2, y2), (0, 0, 255))

    dst = cv2.addWeighted(image, 0.8, lined, 1, 0)
    # Blend the images using the mask
    #dst = cv2.addWeighted(mask, 0.4, image, 0.6, 0.0)

    # Display the resulting frame
    cv2.imshow(title_window, dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 
# After the loop release the cap object
video.release()
# Destroy all the windows
cv2.destroyAllWindows()

'''