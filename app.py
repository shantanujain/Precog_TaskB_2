import numpy as np
import urllib
import json
import cv2
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import dlib
import numpy as np

app = Flask(__name__)

SHAPE_PREDICTOR_PATH = "{base_path}/models/model1.dat".format(base_path=os.path.abspath(os.path.dirname(__file__)))
FACEREC_MODEL_PATH = "{base_path}/models/model2.dat".format(base_path=os.path.abspath(os.path.dirname(__file__)))

NM_PATH = "{base_path}/dataset/narendra_modi/0.jpg".format(base_path=os.path.abspath(os.path.dirname(__file__)))
AK_PATH = "{base_path}/dataset/arvind_kejriwal/0.jpg".format(base_path=os.path.abspath(os.path.dirname(__file__)))

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
face_recognizer = dlib.face_recognition_model_v1(FACEREC_MODEL_PATH)

threshold = 0.58

def compute_distance(known_faces, face):
	x = np.linalg.norm(known_faces - face)
	print (x)
	return (x <= threshold)

# Below function returns the face encodings through neural network
def face_encodings(path_to_image):
	image = cv2.imread(path_to_image)
	detected_faces = face_detector(image, 1)

	shapes = [shape_predictor(image, face) for face in detected_faces]
	return [np.array(face_recognizer.compute_face_descriptor(image, shape, 1)) for shape in shapes]

def face_encodings1(image, face):
	shapes = shape_predictor(image, face)
	# computing the face encodings
	return np.array(face_recognizer.compute_face_descriptor(image, shapes, 1))
 
@app.route('/')
def index():
    return "Hello, World!"

@app.route('/detect', methods=['POST'])
def modi_kejriwal():

	faces = []
	labels = []
 
	if request.method == "POST":
		if request.files.get("input", None) is not None:
			img = get_image(path=request.files.get('input', ''))

		nm_encodings = face_encodings(NM_PATH) 
		ak_encodings = face_encodings(AK_PATH)

		detected_faces = face_detector(img, 1)
		print("Number of faces detected: {}".format(len(detected_faces)))
		for face in detected_faces:

		    test_face_encodings = face_encodings1(img, face)
		    
		    ak = compute_distance(ak_encodings, test_face_encodings)
		    nm = compute_distance(nm_encodings, test_face_encodings)
		    print (ak, nm)
		    
		    if ak:
		    	labels.append(1)
		    elif nm:
		    	labels.append(2)
		    else:
		    	labels.append(0)

		    faces.append([face.left(), face.top(), face.right(), face.bottom()])
 
		response = {"faces": faces, "face_present": len(faces)>0, "labels": labels}
 
	return jsonify(response)
 
def get_image(path):

	data = path.read()

	image = np.asarray(bytearray(data), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	return image


if __name__ == '__main__':
    app.run(debug=True)
