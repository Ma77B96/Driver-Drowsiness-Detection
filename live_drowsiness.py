import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
import os
import numpy as np


face = cv2.CascadeClassifier("./haar/haarcascade_frontalface_alt.xml")
r_eye = cv2.CascadeClassifier("./haar/haarcascade_righteye_2splits.xml")
l_eye = cv2.CascadeClassifier("./haar/haarcascade_lefteye_2splits.xml")

model = tf.keras.models.load_model("./drowiness_new_color.h5")

path = os.getcwd()

cap = cv2.VideoCapture(0) # prima fotocamera del dispositivo 

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

score=0

reye_prediction = [999]
leye_prediction = [999]

ret, _ = cap.read()

if (not ret):
    print("Webcam non disponibile")
    exit(0)

while (cap.isOpened()):
    _, frame = cap.read()

    height,width = frame.shape[:2]

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(frame, 1.1, 15)
    reye = r_eye.detectMultiScale(frame,1.1, 30)
    leye = l_eye.detectMultiScale(frame, 1.1, 30)

    for f in faces:
        cv2.rectangle(frame, (f[0], f[1]), (f[0]+f[2], f[1]+f[3]), (0, 255, 0), 2)


    # occhio destro 
    for r in reye:
        cv2.rectangle(frame, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255,0,0), 2)
        right_eye = frame[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]

        # processamneto del frame prima della predict
        #reye_array = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        reye_array = right_eye / 255
        resized_array = cv2.resize(reye_array, (64,64))
        pred_reye = resized_array.reshape(-1, 64, 64, 3)

        reye_prediction = model.predict(np.round(pred_reye))
        if(np.round(reye_prediction[0])==1):
            lbl='Open' 
        if(np.round(reye_prediction[0])==0):
            lbl='Closed'
        break

    # occhio sinistro
    for l in leye:
        cv2.rectangle(frame, (l[0], l[1]), (l[0]+l[2], l[1]+l[3]), (255,0,0), 2)
        left_eye = frame[l[1]:l[1]+l[3], l[0]:l[0]+l[2]]


        # processamento del frame prima della predict
        #leye_array = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        leye_array = left_eye / 255
        resizedArray = cv2.resize(leye_array, (64,64))
        pred_leye = resizedArray.reshape(-1, 64, 64, 3)

        leye_prediction = model.predict(pred_leye)
        if(np.round(leye_prediction[0])==1):
            lbl='Open' 
        if(np.round(leye_prediction[0])==0):
            lbl='Closed'
        break


    # controllo sulle prediction dei frame relativi all'occhio sinistro e all'occhio destro: se entrambi chiusi incremento lo score,
    # altrimenti lo decremento
    if(np.round(reye_prediction[0]) ==0 and np.round(leye_prediction[0]) ==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)


    # controllo sullo score con soglia fissata a 30
    if(score>20):
        # Il conducente ha sonno: svegliamolo!!
        cv2.putText(frame, "FAI UNA PAUSA CAFFE'!", (frame.shape[1] - 1700, frame.shape[0]-100), cv2.FONT_HERSHEY_DUPLEX, 4.0, (0,0,255), 3)
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            print("FAI UNA PAUSA E PRENDITI UN CAFFE'!!")
        except:  
            pass
        # evidenzio la schermata con un rettangolo spesso e di colore rosso 
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),25) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()







