import pickle

import cv2


def face_recognition(images):
    print('\nStarting Recognizer....')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)

    for img in images:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
        )

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100:
                id = names[id]
                confidence = round(100 - confidence)
                print(confidence)
                if confidence > 80:
                    return id
    return 'unknown'
