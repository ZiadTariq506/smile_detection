import cv2
#                                   our trained AI module
module = cv2.CascadeClassifier("module.xml")

# Load image to our script
img = cv2.imread("Rob.jpg")

# convert it to gray
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# detect faces
face_coordinates = module.detectMultiScale(gray_img)
print("there it is", face_coordinates)
(x, y, w, h) = face_coordinates[0]
# draw the rectangles around the face
#            image   the upper left the bottom right rectangles s' color thick
cv2.rectangle(img, (x, y), (x+w, y+h), (128, 0, 128), 2)

# show the image
cv2.imshow("a Gray Image", img)
# close it when we press any key
cv2.waitKey()
print("Keep Going")