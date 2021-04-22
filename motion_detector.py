#This is a Motion Detection program in Python which uses webcam to detect When a user/object comes in frame and when it leaves the frame
import cv2, time, pandas
from datetime import datetime

fframe=None
status=[None,None]
times=[]
dataframe=pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)								

while True:
    check, frame = video.read()							
    status=0
    grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)	
    grayscale=cv2.GaussianBlur(grayscale,(21,21),0)		

    if fframe is None:
        fframe=grayscale
        continue

    delta=cv2.absdiff(fframe,grayscale)
    threshold=cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]
    threshold=cv2.dilate(threshold, None, iterations=2)

    (_,cnts,_)=cv2.findContours(threshold.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status=1

        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
    status.append(status)

    status=status[-2:]


    if status[-1]==1 and status[-2]==0:
        times.append(datetime.now())
    if status[-1]==0 and status[-2]==1:
        times.append(datetime.now())


    cv2.imshow("grayscale Frame",grayscale)
    cv2.imshow("Delta Frame",delta)
    cv2.imshow("Threshold Frame",threshold)
    cv2.imshow("Color Frame",frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

print(status)
print(times)

for i in range(0,len(times),2):
    dataframe=dataframe.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

dataframe.to_csv("Timings.csv")

video.release()
cv2.destroyAllWindows
