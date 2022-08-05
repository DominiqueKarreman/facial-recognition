
import numpy as np 
import cv2 
import pickle


face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
# profile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_upperbody.xml")
check = input("check")
print(check)

cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

      
        
      
        if check == "yes":  
            img_item = f"scan N--{count}.png"
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
    count += 1

cap.release()
cv2.destroyAllWindows()