import cv2
import os
import numpy as np
import sqlite3
import time
import xlsxwriter
from PIL import Image


#----------------------------------------------------------------------------------#
#----------------------------TO REGISTER NEW USER----------------------------------# 
#----------------------------------------------------------------------------------#

def register_user():
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam=cv2.VideoCapture(0);
    def insert(Id,Name):
        conn=sqlite3.connect("StudentData.db")
        cmd="SELECT * FROM Data WHERE ID="+str(Id)
        cursor=conn.execute(cmd)
        isRecord=0
        for row in cursor:
            isRecord=1
        if(isRecord==1):
            cmd="UPDATE Data set Name="+str(Name)+"WHERE ID="+str(Id)
        else:
            cmd="INSERT INTO Data(ID,NAME) Values("+str(Id)+","+str(Name)+")"
        conn.execute(cmd)
        conn.commit()
        conn.close()                
    id=raw_input('enter user id')
    name=raw_input('enter the name')
    insert(id,name)
    sampleNum=0
    while(True):
        ret,img=cam.read();
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
            sampleNum=sampleNum+1
            cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
            cv2.waitKey(100);
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Face",img);
        cv2.waitKey(1)
        if(sampleNum>20):
            break
    cam.release();
    cv2.destroyAllWindows()

#----------------------------------------------------------------------------------#
#----------------------------TRAINER FOR HAAR CLASSIFIER---------------------------# 
#----------------------------------------------------------------------------------#
    
def trainer():
    recognizer=cv2.createLBPHFaceRecognizer();
    path='dataSet'
    imagePaths=[]
    def getImageswithID(path):
        for f in os.listdir(path):
            if f!='desktop.ini' and f!='Thumbs.db':
                imagePaths.append(os.path.join(path,f))
        faces=[]
        IDs=[]
        for imagepath in imagePaths:
            faceImg=Image.open(imagepath).convert('L');
            faceNp=np.array(faceImg,'uint8')
            ID=int(os.path.split(imagepath)[-1].split('.')[1])
            faces.append(faceNp)
            print ID
            IDs.append(ID)
            cv2.imshow("trainning",faceNp)
            cv2.waitKey(10)
        return np.array(IDs),faces
    IDs,faces=getImageswithID(path)
    recognizer.train(faces,np.array(IDs))
    recognizer.save('recognizer/trainningData.yml')
    cv2.destroyAllWindows()

#----------------------------------------------------------------------------------#
#----------------------------TO DETECT USER FACES----------------------------------# 
#----------------------------------------------------------------------------------#
    
def detect():
    roll_no=set()
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam=cv2.VideoCapture(0);
    rec=cv2.createLBPHFaceRecognizer();
    rec.load("recognizer/trainningData.yml")
    id=0
    def write_to_excel():
        col=['A1','A2','A3','A4''A5','A6','A7','A8']
        col2=['B1','B2','B3','B4''B5','B6','B7','B8']
        k=0
        workbook=xlsxwriter.Workbook('attendance.xlsx')
        worksheet=workbook.add_worksheet()
        for i in roll_no:
            profile=getProfile(i)
            worksheet.write(col[k],profile[1])
            worksheet.write(col2[k],"Present")
            k=k+1
    
      
    def getProfile(id):
        conn=sqlite3.connect("StudentData.db")
        cmd="SELECT * FROM Data WHERE ID="+str(id)
        cursor=conn.execute(cmd)
        profile=None
        for row in cursor:
            profile=row
        conn.close()
        return profile
    font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,6,1,0,1)
    time.sleep(2)
    while(True):
        ret,img=cam.read();
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            id,conf=rec.predict(gray[y:y+h,x:x+w])
            if conf<50:
                roll_no.add(id)
                print(roll_no)
                profile=getProfile(id)
                if(profile!=None):
                    cv2.cv.PutText(cv2.cv.fromarray(img),profile[1],(x,y+h),font,255);      
        cv2.imshow("Face",img);
        write_to_excel()
        if(cv2.waitKey(1)==ord('q')):
            break
    write_to_excel()
    cam.relese();
    cv2.destroyAllWindows()
