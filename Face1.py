from datetime import datetime

import cv2
import face_recognition
import os

import numpy as np

path = "pic2"
images = [cv2.imread(f"{path}/{item}") for item in os.listdir(path)]
classNames = [os.path.splitext(item)[0] for item in os.listdir(path)]


def EnCode(images):
    list_encode = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_encode = face_recognition.face_encodings(img)[0]
        list_encode.append(img_encode)
    return list_encode


def ExportToExcel(name):
    with open('excelFaceAuthencation.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            date_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{date_string}')


encodeKnows = EnCode(images)

cap = cv2.VideoCapture("video-test.mp4")

while True:
    ret, frame_flip = cap.read()
    # frame_flip = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
    locFaces = face_recognition.face_locations(frame_flip)
    encodeFaces = face_recognition.face_encodings(frame_flip)

    for encodeFace, locFace in zip(encodeFaces, locFaces):
        matches = face_recognition.compare_faces(encodeKnows, encodeFace)
        faceDistance = face_recognition.face_distance(encodeKnows, encodeFace)

        matchIndex = np.argmin(faceDistance)

        if faceDistance[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
            ExportToExcel(name)
        else:
            name = "UnKnow"

        y1, x2, y2, x1 = locFace
        cv2.rectangle(frame_flip, (x1, y1), (x2, y2), (0, 255, 245), 2)
        cv2.putText(frame_flip, name, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Camera", frame_flip)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
