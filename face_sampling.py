import cv2
import numpy as np
from PIL import Image
import pickle
import os


def face_sampling(images, name):
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)
    names.append(name)
    id = names.index(name)
    cropped_faces = []

    # Initialize individual sampling face count
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped_faces.append(gray[y:y + h, x:x + w])
    with open('names.pkl', 'wb') as f:
        pickle.dump(names, f)

    # Do a bit of cleanup
    print("Your Face has been registered as {}\n\nExiting Sampling Program".format(name.upper()))
    return cropped_faces
