import numpy as np
import matplotlib.pyplot as plt
import cv2

# img = np.zeros((512,512))
# cv2.circle(img,(256,256),radius=200,color=[255,255,255],thickness=10)
# cv2.circle(img,(250,360),radius=50,color=[255,255,255],thickness=-1)
# cv2.circle(img,(380,300),radius=30,color=[255,255,255],thickness=-1)
# cv2.circle(img,(170,150),radius=25,color=[255,255,255],thickness=-1)
# plt.imshow(img,cmap='gray')

# cv2.imwrite('bowling-ball.png', img)
T_factor = 0.97
T = 4
T_min = 10**(-14)
yAxis = []

while T > T_min:
    yAxis.append(T)
    T = T*T_factor

xAxis = np.arange(0, len(yAxis), 1)

plt.plot(xAxis, yAxis)
plt.xlabel('Number of iterations')
plt.ylabel('Value of T')
plt.show()