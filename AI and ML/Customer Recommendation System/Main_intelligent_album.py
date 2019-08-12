
# coding: utf-8

# Loading libraries
import glob
import dlib
import cv2
import pickle
import random
import facevec
import numpy as np

# Loading pickle file

pickle_in = open("intelligent_album_snist.pickle","rb")
classifier = pickle.load(pickle_in)

# Person names in album

name = ['Sai Kumar: ', 'Sai Teja: ', 'Unknown: ']
font = cv2.FONT_HERSHEY_DUPLEX

# Function for intelligent Album



def intelligent_album(image):
    persons = []
    orig = cv2.imread(image)
    img = cv2.resize(orig, (0, 0), fx=0.5, fy=0.5)
    faces = facevec.detector(img,1)

    if len(faces) > 0:
        for i in range(len(faces)):

            f = faces[i]
            shapes = facevec.predictor(img,f)
            face_descriptor = facevec.face_model.compute_face_descriptor(img, shapes)
            face_descriptor = np.array(face_descriptor)
            descriptor = face_descriptor.reshape(1,-1)
            gender = classifier.predict_proba(descriptor)

            if int(gender[0][gender.argmax()] * 100) >= 85:
                person = name[gender.argmax()]
                album = gender.argmax()


            else:
                person = 'Unknown: '
                album = 2

            cv2.rectangle(img,(f.left(), f.top()), (f.right(),f.top()-20),(0,255,0), -1)
            cv2.rectangle(img, (f.left(), f.top()), (f.right(), f.bottom()),(0,255,0),1)
            cv2.putText(img,   person +
                        str(int(gender[0][gender.argmax()] * 100))+'%', (f.left(),f.top()), font, 0.6, (255,255,255), 0)
            
            
            
            persons.append(album)
            
            if i == len(faces)-1:
                print('entered')
                persons.sort()
                print(persons)
                if persons[0] == 2:

                    num = len(glob.glob('./Album/Others/*'))
                    num1 = num+1
                    cv2.imwrite('./Album/Others/other_'+str(num)+'.jpg', orig)
                    cv2.imwrite('./Album/Others/other_'+str(num1)+'.jpg', img)
                    print('Saved Sucessfully in Album/Others !!! :)')

            if album == 0:
                num = len(glob.glob('./Album/SaiKumar/*'))
                num1 = num+1
                cv2.imwrite('./Album/SaiKumar/kumar_'+str(num)+'.jpg', orig)
                cv2.imwrite('./Album/SaiKumar/kumar_'+str(num1)+'.jpg', img)
                print('Saved Sucessfully in Album/SaiKumar !!! :)')

            elif album == 1:
                num = len(glob.glob('./Album/SaiTeja/*'))
                num1 = num+1
                cv2.imwrite('./Album/SaiTeja/teja_'+str(num)+'.jpg', orig)
                cv2.imwrite('./Album/SaiTeja/teja_'+str(num1)+'.jpg', img)
                print('Saved Sucessfully in Album/SaiTeja !!! :)')
                
            

        
    else:
        album = 2
        num = len(glob.glob('./Album/Others/*'))
        cv2.imwrite('./Album/Others/other_'+str(num)+'.jpg', orig)
        print('Saved Sucessfully in Album/Others !!! :)')
        
        

# Loading images from Folder Images

images = glob.glob('./Images/*')
for i in range(len(images)):
    intelligent_album(images[i])

