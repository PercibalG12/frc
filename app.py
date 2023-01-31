import cv2
from random import randrange 

# load pretrained data on faces from open cv2 files
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam= cv2.VideoCapture(0)

while True:
    #raed the current frame
    successful_frame_read, frame= webcam.read()
    #Convert image to grayscale
    grayscaled_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces 
    face_cordinates=trained_face_data.detectMultiScale(grayscaled_img)
    for (x,y,w,h) in face_cordinates:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (randrange(128,256),randrange(128,256),randrange(128,256) ), 10)
    cv2.imshow('Clever Programmer Face Detect',frame)
    key = cv2.waitKey(1)

    #stop if q is pressed
    if key == 81 or key ==113 or key ==66 or key ==98:
        break
webcam.release()
# #detect faces 
# face_cordinates=trained_face_data.detectMultiScale(grayscaled_img)

# #draw rectangles around the faces 
# for (x,y,w,h) in face_cordinates:
#     cv2.rectangle(webcam, (x,y),(x+w, y+h), (randrange(128,256),randrange(128,256),randrange(128,256) ), 10)
#     print("drawing")
# # (x,y,w,h) = face_cordinates[0]
# # cv2.rectangle(img, (x,y),(x+w, y+h), (0, 255, 0), 2)

# # print(face_cordinates)
# cv2.imshow('Clever Programmer Face Detect',webcam_filename)
# cv2.waitKey()



 

