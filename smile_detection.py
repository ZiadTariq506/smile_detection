import cv2
from random import randrange
#                                   our trained AI module
module = cv2.CascadeClassifier("module.xml")

# Load image to our script
img = cv2.imread("2.webp")

# convert it to gray
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# detect faces
face_coordinates = module.detectMultiScale(gray_img)
print("there it is", face_coordinates)
# draw on all the faces
for (x, y, w, h) in face_coordinates:
    # draw the rectangles around the face
    # image the upper left the bottom right rectangles s' color thick
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

# show the image
cv2.imshow("a Gray Image", img)
# close it when we press any key
cv2.waitKey()
print("script is done :)")
