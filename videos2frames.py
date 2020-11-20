import os
import cv2

root='/Users/osvaldocartagena/logs/videos'

ls=os.listdir(root)

for vname in ls:
    k=0
    cap = cv2.VideoCapture(root+'/'+vname)
    print(root+'/'+vname)
        
    vname=vname[:-4]
    while(cap.isOpened()):
        print(str(k))
        ret, frame = cap.read()
        if ret==False:
            break
        if k%60==0:
            cv2.imwrite('/Users/osvaldocartagena/frames/'+vname+str(int(k/10))+'.jpg', frame)
            print('frames/'+vname+str(int(k/60))+'.png')
        k=k+1

    cap.release()
    


    
