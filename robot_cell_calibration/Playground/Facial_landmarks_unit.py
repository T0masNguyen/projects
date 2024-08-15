# find help under:
# https://www.youtube.com/watch?v=Yg6bFRnOSbs
# import the cv2 library
import csv
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


img_base = cv2.imread('all_landmarks_.png', 1)
img = img_base.copy()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
results = face_mesh.process(img)
landmarks=results.multi_face_landmarks[0]


xs = [];ys = [];zs = []
i=0
#----------------------------------------------------------------------write landmarks coord to file (image coordsystem)
outputFileFD = open('Test_Landmarks.txt', 'w')
writer = csv.writer(outputFileFD, delimiter="\t")
writer.writerow(['id', 'LM_x', 'LM_y'])
LM_X = np.empty(469)
LM_Y = np.empty(469)

img2 = img_base.copy()
for landmark in landmarks.landmark:

    x=landmark.x
    y=landmark.y
    z=landmark.z

    xs.append(x)
    ys.append(y)
    zs.append(z)

    relative_x = int(x * img.shape[1])
    relative_y = int(y * img.shape[0])

    LM_X[i] = relative_x
    LM_Y[i] = relative_y

    cv2.circle(img2, (relative_x, relative_y), radius=6, color=(0, 255, 0), thickness=5)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (relative_x+15, relative_y+30)
    fontScale = 2
    color = (0, 255, 0)
    thickness = 5
    cv2.putText(img2, str(i), org, font, fontScale, color,  thickness, cv2.LINE_AA)

    #fig = plt.figure(figsize=(15, 15))
    image_name = "Number"
    image_end = ".png"
    image_new = image_name+str(i)+image_end

    img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.imsave(image_new, arr=img_rgb, format='png')
    img2 = img_base.copy()

    arrayLM = np.array([i, LM_X[i], LM_Y[i]])
    i = i + 1

    writer.writerow(arrayLM)

outputFileFD.flush()
outputFileFD.close()
del writer


img_rgb2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img[:,:,::-1])
plt.show()
plt.imsave("all_landmarks.png", arr=img_rgb2, format='png')

#--------------------------------------plot 3d points
fig = plt.figure(figsize = (15,15))
fig = plt.figure()
ax = Axes3D(fig)
#ax.text(xs,ys,zs, '%s', size=3, zorder=1, color='k')
projection = ax.scatter(xs,ys,zs,color='green')

plt.show()
