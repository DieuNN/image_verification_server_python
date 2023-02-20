import cv2
import numpy as np
from PIL import Image
import os


def face_learning(cropped_faces):
    # Path for face image database

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # function to get the images and label data
    def get_images_and_labels():

        faceSamples = []
        ids = []

        for face in cropped_faces:
            img_numpy = np.array(face, 'uint8')

            id = int(os.getenv('ID'))
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        return faceSamples, ids

    print("\nTraining for the faces has been started. It might take a while.\n")
    faces, ids = get_images_and_labels()
    recognizer.train(faces, np.array(ids))
    os.environ['ID'] = str(int(os.getenv('ID')) + 1)

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')

    # Print the numer of faces trained and end program
    print("{0} faces trained. Exiting Training Program".format(len(np.unique(ids))))
