import requests
import cv2
import os
import sys
import json
 
image_path = sys.argv[1]

url = "http://localhost:5000/detect"

image = cv2.imread(image_path)
payload = {"input": open(image_path, "rb")}
response = requests.post(url, files=payload).json()

# print "Face Present: " + str(response["face_present"])
# print "Narendra Modi: " + str(response["modi"])
# print "Arvind Kejriwal: " + str(response["kejriwal"])
 
for (face, label) in zip(response["faces"], response["labels"]):
	cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)
	if label == 1:
		cv2.putText(image, "Kejriwal", (face[0], face[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
	elif label == 2:
		cv2.putText(image, "Modi", (face[0], face[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# small = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
cv2.imshow("Faces Found", image)
cv2.waitKey(0)