import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image
# img_one = cv2.imread('1.JPG')

# b,g,r = cv2.split(img_one)
# img_one = cv2.merge((b,g,r))
# img_one[:,:200,2] = 0
# # show image
# cv2.imshow('img_one', img_one)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# show plot
# plt.imshow(img_one, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

# print(img_one.shape)
img = cv2.imread('dataset/D.jpg', 0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
