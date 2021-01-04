import os
import cv2

root='/Users/osvaldocartagena/logs/videos' #path a la carpeta que contiene los videos
#la carpeta logs esta disponible en https://github.com/duckietown/logs

ls=os.listdir(root) #lista con los nombres de los videos

for vname in ls: #toma el nombre de un vidoe
    k=0
    cap = cv2.VideoCapture(root+'/'+vname) #crea el objeto video capture
        
    vname=vname[:-4] #remueve la extension .mp4 del video
    
    while(cap.isOpened()): 
        ret, frame = cap.read() #abre un frame del video
        if ret==False:
            break
        if k%60==0: #guarda 1 de cada 60 frames
            cv2.imwrite('/Users/osvaldocartagena/frames/'+vname+str(int(k/60))+'.jpg', frame) #guarda frame en carpeta frames
        k=k+1

    cap.release() #borra el objeto video capture
    


    
