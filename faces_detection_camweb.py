import cv2
from random import randrange

#                                   our trained AI module
module = cv2.CascadeClassifier("module.xml")

# Load image to our script
# img = cv2.imread("2.webp")

# take the video from the webcam
webcam = cv2.VideoCapture(0)

while True:
    # is it able to get the frame and the frame
    able_to_get_frame, frame = webcam.read()

# convert the frame to gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # detect faces
    face_coordinates = module.detectMultiScale(gray_frame)
    # print("there it is", face_coordinates)

    # draw on all the faces
    for (x, y, w, h) in face_coordinates:
        # draw the rectangles around the face
        # image the upper left the bottom right rectangles s' color thick
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
    # play the video
    cv2.imshow("Detecting faces", frame)
    # wait the key to close (this must choose a number to delay or it will stand in way of capture the video
    cv2.waitKey(1)
    print("script is working :)")