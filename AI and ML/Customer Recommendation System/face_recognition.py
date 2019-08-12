
# coding: utf-8

# In[1]:


import glob
import dlib
import cv2
import pickle
import random
import facevec
import numpy as np


# In[2]:


pickle_in = open("intelligent_album_snist.pickle","rb")
classifier = pickle.load(pickle_in)


# In[7]:


name = ['Sai Kumar: ', 'Sai Teja: ', 'Unknown: ']


# In[4]:


font = cv2.FONT_HERSHEY_DUPLEX


# In[9]:


cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    faces = facevec.detector(img,1)
    
    if len(faces) > 0:
        for i in range(len(faces)):
            
            f = faces[i]
            shapes = facevec.predictor(img,f)
            face_descriptor = facevec.face_model.compute_face_descriptor(img, shapes)
            face_descriptor = np.array(face_descriptor)
            descriptor = face_descriptor.reshape(1,-1)
            gender = classifier.predict_proba(descriptor)
            
            if int(gender[0][gender.argmax()] * 100) >= 75:
                person = name[gender.argmax()]
                album = gender.argmax()
                
            else:
                person = 'Unknown: '
                album = 2


            cv2.rectangle(img,(f.left(), f.top()), (f.right(),f.top()-20),(0,255,0), -1)
            cv2.rectangle(img, (f.left(), f.top()), (f.right(), f.bottom()),(0,255,0),1)
            cv2.putText(img,   person +
                        str(int(gender[0][gender.argmax()] * 100))+'%', (f.left(),f.top()), font, 0.6, (255,255,255), 0)


    cv2.imshow('image',img)
    if cv2.waitKey(41) & 0xff == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

