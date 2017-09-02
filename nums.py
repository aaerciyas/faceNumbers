import face_recognition
import cv2
import os, os.path
import numpy as np
from threading import Thread
import queue

video_capture = cv2.VideoCapture(0)

def takePhoto():
    ret,image = video_capture.read()
    small_image = cv2.resize(image , (0, 0), fx=0.5, fy=0.5)
    DIR = "/home/ahmet/Templates/cVision/faceDetect/images"
    numOfFiles = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    i = numOfFiles
    cv2.imwrite("images/" + str(i) + ".jpg",small_image)
    #print("I am takePhoto function")

def compareWithImages(cam_face_encodings):
    DIR = "/home/ahmet/Templates/cVision/faceDetect/images"
    numOfFiles = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    for i in range(numOfFiles):
        image = face_recognition.load_image_file("images/" + str(i) + ".jpg")
        image_Encoding = face_recognition.face_encodings(image)[0]
        for cam_face_encoding in cam_face_encodings:
            match = face_recognition.compare_faces([image_Encoding], cam_face_encoding)
        match = bool(match[0])
        #print(match)

        if match:
            #print("eslesme saglandi")
            return ("True",i)
    return ("False",i)


def faceDetect(selam,q):
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    if face_locations:
        comp = compareWithImages(face_encodings)
        compare, number = comp
        if compare == "False":
            takePhoto()
        #print(number)
        for (top, right, bottom, left), name in zip(face_locations, str(number)):
            #print("this is my for loop")
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        q.put(frame)


while True:
    q = queue.Queue()
    t = Thread(target= faceDetect, args=("selan",q)).start()
    frame = q.get()
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()