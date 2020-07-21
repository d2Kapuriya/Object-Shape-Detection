
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio
import cv2

#Load image into variable and display it
lion = imageio.imread('circle4.png') # PasteQqq address of image
plt.imshow(lion, cmap = plt.get_cmap('gray'))
plt.show()

image = imageio.imread('circle4.png') # Paste address of image
# plt.imshow(image, cmap = plt.get_cmap('gray'))
# plt.show()


# Convert color image to grayscale to help extraction of edges and plot it
lion_gray = np.dot(lion[...,:3], [0.299, 0.587, 0.114])
#lion_gray = lion_gray.astype('int32')
# plt.imshow(lion_gray, cmap = plt.get_cmap('gray'))
# plt.show()


# Blur the grayscale image so that only important edges are extracted and the noisy ones ignored
lion_gray_blurred = ndimage.gaussian_filter(lion_gray, sigma=1.4) # Note that the value of sigma is image specific
# plt.imshow(lion_gray_blurred, cmap = plt.get_cmap('gray'))
# plt.show()


# Apply Sobel Filter using the convolution operation
# Note that in this case I have used the filter to have a maximum amgnitude of 2, but it can also be changed to other numbers for aggressive edge extraction
# For eg [-1,0,1], [-5,0,5], [-1,0,1]
def SobelFilter(img, direction):
    if (direction == 'x'):
        Gx = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
        Res = ndimage.convolve(img, Gx)
        # Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
    if (direction == 'y'):
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
        Res = ndimage.convolve(img, Gy)
        # Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)

    return Res


# Normalize the pixel array, so that values are <= 1
def Normalize(img):
    #img = np.multiply(img, 255 / np.max(img))
    img = img/np.max(img)
    return img


# Apply Sobel Filter in X direction
gx = SobelFilter(lion_gray_blurred, 'x')
gx = Normalize(gx)
# plt.imshow(gx, cmap = plt.get_cmap('gray'))
# plt.show()


# Apply Sobel Filter in Y direction
gy = SobelFilter(lion_gray_blurred, 'y')
gy = Normalize(gy)
# plt.imshow(gy, cmap = plt.get_cmap('gray'))
# plt.show()


# Apply the Sobel Filter using the inbuilt function of scipy, this was done to verify the values obtained from above
# Also differnet modes can be tried out for example as given below:
#dx = ndimage.sobel(lion_gray_blurred, axis=1, mode='constant', cval=0.0)  # horizontal derivative
#dy = ndimage.sobel(lion_gray_blurred, axis=0, mode='constant', cval=0.0)  # vertical derivative

dx = ndimage.sobel(lion_gray_blurred, axis=1) # horizontal derivative
dy = ndimage.sobel(lion_gray_blurred, axis=0) # vertical derivative


# Plot the derivative filter values obtained using the inbuilt function
# plt.subplot(121)
# plt.imshow(dx, cmap = plt.get_cmap('gray'))
# plt.subplot(122)
# plt.imshow(dy, cmap = plt.get_cmap('gray'))
# plt.show()

# Calculate the magnitude of the gradients obtained
Mag = np.hypot(gx,gy)
Mag = Normalize(Mag)
# plt.imshow(Mag, cmap = plt.get_cmap('gray'))
# plt.show()


# Calculate the magnitude of the gradients obtained using the inbuilt function, again done to verify the correctness of the above value
mag = np.hypot(dx,dy)
mag = Normalize(mag)
# plt.imshow(mag, cmap = plt.get_cmap('gray'))
# plt.show()


# Calculate direction of the gradients
Gradient = np.degrees(np.arctan2(gy,gx))

# Calculate the direction of the gradients obtained using the inbuilt sobel function
gradient = np.degrees(np.arctan2(dy,dx))


# Do Non Maximum Suppression with interpolation to get a better estimate of the magnitude values of the pixels in the gradient direction
# This is done to get thin edges
def NonMaxSupWithInterpol(Gmag, Grad, Gx, Gy):
    NMS = np.zeros(Gmag.shape)

    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if ((Grad[i, j] >= 0 and Grad[i, j] <= 45) or (Grad[i, j] < -135 and Grad[i, j] >= -180)):
                yBot = np.array([Gmag[i, j + 1], Gmag[i + 1, j + 1]])
                yTop = np.array([Gmag[i, j - 1], Gmag[i - 1, j - 1]])
                x_est = np.absolute(Gy[i, j] / Gmag[i, j])
                if (Gmag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Gmag[i, j] >= (
                        (yTop[1] - yTop[0]) * x_est + yTop[0])):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0
            if ((Grad[i, j] > 45 and Grad[i, j] <= 90) or (Grad[i, j] < -90 and Grad[i, j] >= -135)):
                yBot = np.array([Gmag[i + 1, j], Gmag[i + 1, j + 1]])
                yTop = np.array([Gmag[i - 1, j], Gmag[i - 1, j - 1]])
                x_est = np.absolute(Gx[i, j] / Gmag[i, j])
                if (Gmag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Gmag[i, j] >= (
                        (yTop[1] - yTop[0]) * x_est + yTop[0])):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0
            if ((Grad[i, j] > 90 and Grad[i, j] <= 135) or (Grad[i, j] < -45 and Grad[i, j] >= -90)):
                yBot = np.array([Gmag[i + 1, j], Gmag[i + 1, j - 1]])
                yTop = np.array([Gmag[i - 1, j], Gmag[i - 1, j + 1]])
                x_est = np.absolute(Gx[i, j] / Gmag[i, j])
                if (Gmag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Gmag[i, j] >= (
                        (yTop[1] - yTop[0]) * x_est + yTop[0])):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0
            if ((Grad[i, j] > 135 and Grad[i, j] <= 180) or (Grad[i, j] < 0 and Grad[i, j] >= -45)):
                yBot = np.array([Gmag[i, j - 1], Gmag[i + 1, j - 1]])
                yTop = np.array([Gmag[i, j + 1], Gmag[i - 1, j + 1]])
                x_est = np.absolute(Gy[i, j] / Gmag[i, j])
                if (Gmag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Gmag[i, j] >= (
                        (yTop[1] - yTop[0]) * x_est + yTop[0])):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0

    return NMS


# This is also non-maxima suppression but without interpolation i.e. the pixel closest to the gradient direction is used as the estimate
def NonMaxSupWithoutInterpol(Gmag, Grad):
    NMS = np.zeros(Gmag.shape)
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Grad[i,j] >= -22.5 and Grad[i,j] <= 22.5) or (Grad[i,j] <= -157.5 and Grad[i,j] >= 157.5)):
                if((Gmag[i,j] > Gmag[i,j+1]) and (Gmag[i,j] > Gmag[i,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 22.5 and Grad[i,j] <= 67.5) or (Grad[i,j] <= -112.5 and Grad[i,j] >= -157.5)):
                if((Gmag[i,j] > Gmag[i+1,j+1]) and (Gmag[i,j] > Gmag[i-1,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 67.5 and Grad[i,j] <= 112.5) or (Grad[i,j] <= -67.5 and Grad[i,j] >= -112.5)):
                if((Gmag[i,j] > Gmag[i+1,j]) and (Gmag[i,j] > Gmag[i-1,j])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 112.5 and Grad[i,j] <= 157.5) or (Grad[i,j] <= -22.5 and Grad[i,j] >= -67.5)):
                if((Gmag[i,j] > Gmag[i+1,j-1]) and (Gmag[i,j] > Gmag[i-1,j+1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0

    return NMS


# Get the Non-Max Suppressed output
NMS = NonMaxSupWithInterpol(Mag, Gradient, gx, gy)
NMS = Normalize(NMS)
# plt.imshow(NMS, cmap = plt.get_cmap('gray'))
# plt.show()


# Get the Non-max suppressed output on the same image but using the image using the inbuilt sobel operator
nms = NonMaxSupWithInterpol(mag, gradient, dx, dy)
nms = Normalize(nms)
# plt.imshow(nms, cmap = plt.get_cmap('gray'))
# plt.show()


# Double threshold Hysterisis
# Note that I have used a very slow iterative approach for ease of understanding, a faster implementation using recursion can be done instead
# This recursive approach would recurse through every strong edge and find all connected weak edges
def DoThreshHyst(img):
    highThresholdRatio = 0.2
    lowThresholdRatio = 0.15
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    x = 0.1
    oldx = 0

    # The while loop is used so that the loop will keep executing till the number of strong edges do not change, i.e all weak edges connected to strong edges have been found
    while (oldx != x):
        oldx = x
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if (GSup[i, j] > highThreshold):
                    GSup[i, j] = 1
                elif (GSup[i, j] < lowThreshold):
                    GSup[i, j] = 0
                else:
                    if ((GSup[i - 1, j - 1] > highThreshold) or
                            (GSup[i - 1, j] > highThreshold) or
                            (GSup[i - 1, j + 1] > highThreshold) or
                            (GSup[i, j - 1] > highThreshold) or
                            (GSup[i, j + 1] > highThreshold) or
                            (GSup[i + 1, j - 1] > highThreshold) or
                            (GSup[i + 1, j] > highThreshold) or
                            (GSup[i + 1, j + 1] > highThreshold)):
                        GSup[i, j] = 1
        x = np.sum(GSup == 1)

    GSup = (GSup == 1) * GSup  # This is done to remove/clean all the weak edges which are not connected to strong edges

    return GSup



# The output of canny edge detection
Final_Image = DoThreshHyst(NMS)
plt.imshow(Final_Image, cmap = plt.get_cmap('gray'))
# plt.show()


# The output of canny edge detection using the inputs obtaind using the inbuilt sobel operator
# Notice that the output here looks better than the one above, this might be because of the low magnitude of filter value used in our implementation of the Sobel Operator
# Changing the filter to a higher value leads to more aggressive edge extraction and thus a better output.
final_image = DoThreshHyst(nms)
# plt.imshow(final_image, cmap = plt.get_cmap('gray'))
# plt.show()

# print(final_image.dtype)
final_image = np.uint8(final_image)
# print(final_image.dtype)

# ret, thresh1 = cv2.threshold(final_image, 127, 255, cv2.THRESH_BINARY)



# success = 1
# while success:

font = cv2.FONT_HERSHEY_COMPLEX

contours, hierarchy = cv2.findContours(final_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color = (0, 0, 255)
#
# image1 = imageio.imread('circle4.png') # Paste address of image
# plt.imshow(image1, cmap = plt.get_cmap('gray'))
# plt.show()

for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.00001*cv2.arcLength(cnt, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]


            cv2.drawContours(image, [approx], 0, (255, 0, 0), 5)

            if len(approx) == 3:
                cv2.putText(image, "Triangle", (x, y), font, 1,(0,0,255), (0))

            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)

                cv2.putText(image, "Square", (x, y), font, 1,(0,0,255), (0)) if ar >= 0.95 and ar <= 1.05 else cv2.putText(image, "Rectangle", (x, y), font, 1,(0,0,255), (0))
            elif len(approx) == 5:

                cv2.putText(image, "Pentagon", (x, y), font, 1,(0,0,255), (0))
            elif len(approx) == 6:

                cv2.putText(image, "Hexagon", (x, y), font, 1,(0,0,255), (0))
            elif len(approx) == 7:

                cv2.putText(image, "Septagon", (x, y), font, 1,(0,0,255), (0))
            elif len(approx) == 8:

                cv2.putText(image, "Octagon", (x, y), font, 1,(0,0,255), (0))
            elif 8 < len(approx) < 13:

                cv2.putText(image, "Ellipse", (x, y), font, 1,(0,0,255), (0))
            elif 13 < len(approx) < 30:

                cv2.putText(image, "Circle", (x, y), font, 1,(0,0,255), (0))

# plt.imshow(image)
# plt.show()
cv2.imshow("shap",image)


# cv2.imshow("Threshold", threshold)
key = cv2.waitKey(0)

cv2.destroyAllWindows()
