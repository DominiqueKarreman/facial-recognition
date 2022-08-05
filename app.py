
import numpy as np 
import cv2 
import pickle


face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
# profile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_upperbody.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray) 
        if conf >= 45 and conf <= 85:
            # print(id_)
            # print(labels[id_], conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0, 255, 255)
            stroke = 2
            conf = round(conf, 0)
            textOut = f"{name} -- {conf}"
            cv2.putText(frame, textOut, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)
        img_item = "my-color.png"
        cv2.imwrite(img_item, roi_color)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
    # profiles = profile_cascade.detectMultiScale(gray)
    # for (x,y,w,h) in profiles:
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = frame[y:y+h, x:x+w]
    #     img_item = "my-image-profile.png"
    #     cv2.imwrite(img_item, roi_gray)
    #     img_item = "my-color-profile.png"
    #     cv2.imwrite(img_item, roi_color)
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 128, 255), 2)


    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()