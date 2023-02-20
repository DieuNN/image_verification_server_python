import os
import glob
import shutil

import cv2
from dotenv import load_dotenv
from flask import Flask, request, jsonify

import face_recognition
import face_sampling
from face_learning import face_learning

app = Flask(__name__)
load_dotenv()


@app.route('/register', methods=["POST"])
def register():
    # Get name and images from uploads route
    name = request.form['name'].replace(" ", "")
    print(name)
    images = request.files.getlist('images')
    print(name)
    # temporary save them to a folder
    for image in images:
        image.save(os.path.join("uploads", image.filename))
    # reassign variable, read images from uploads folder
    images = []
    for file_name in os.listdir('uploads'):
        image = cv2.imread(os.path.join("uploads", file_name))
        images.append(image)
    # cropping images
    cropped_faces = face_sampling.face_sampling(images, name)
    # trainning
    face_learning(cropped_faces)
    # delete all images in uploads
    for f in os.listdir('uploads'):
        os.remove(os.path.join('uploads', f))

    # wanna save cropped images?
    # count = 0
    # for face in cropped_faces:
    #     cv2.imwrite("cropped/" + str(count) + ".jpg", face)
    #     count += 1

    return jsonify({'message': 'register successful'})


@app.route('/login', methods=["POST"])
def login():
    # Get name and images from uploads route
    name = request.form['name']
    images = request.files.getlist('images')
    # temporary save them to a folder
    for image in images:
        image.save(os.path.join("uploads", image.filename))
    # reassign variable, read images from uploads folder
    images = []
    for file_name in os.listdir('uploads'):
        image = cv2.imread(os.path.join("uploads", file_name))
        images.append(image)
    recognized_name = face_recognition.face_recognition(images)
    for f in os.listdir('uploads'):
        os.remove(os.path.join('uploads', f))
    return jsonify({'name': recognized_name})


if __name__ == '__main__':
    app.run()
    # cam = cv2.VideoCapture(0)
    # cam.set(3, 640)  # set video width
    # cam.set(4, 480)
    # face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # count = 0
    # while True:
    #     ret, img = cam.read()
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     faces = face_detector.detectMultiScale(gray, 1.3, 5)
    #
    #     for (x, y, w, h) in faces:
    #         # Save the captured image into the datasets folder
    #         cv2.imwrite("dataset/" + str(count) + ".jpg", img)
    #
    #         cv2.imshow('image', img)
    #         count += 1
    #     k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    #     if k == 27:
    #         break
    #     elif count >= 80:  # Take 80 face sample and stop video
    #         break

    # images = []
    # for file_name in os.listdir("dataset"):
    #     image = cv2.imread(os.path.join("dataset", file_name))
    #     if image is not None:
    #         images.append(image)
    # cropped_faces = face_sampling.face_sampling(images, "dieu")
    # face_learning(cropped_faces)
    # count = 0
    # for face in cropped_faces:
    #     cv2.imwrite("cropped/" + str(count) + ".jpg", face)
    #     count+=1
