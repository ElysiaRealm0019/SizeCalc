import cv2
import sys

img_path = sys.argv[1]
cascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24))

print("Faces found:", faces)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
cv2.imwrite("amiya_faces.png", img)
