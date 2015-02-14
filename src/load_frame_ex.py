import cv2
import numpy as np



# object to get frames from videos
vc = cv2.VideoCapture() 

# open file
video1 = vc.open('Desktop/UCF101/UCF-101/YoYo/v_YoYo_g25_c04.avi') 

# get a frame

r, frame = video1.read()

# frame is now an image
# resize to fit it or do whatever image stuff (pillow Image class)

frame.resize((224,224,3)) # resize example

#show the image
cv2.imshow(frame)

# make numpy array
frame_matrix = np.asarray(frame)
