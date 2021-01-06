import sys
import os

#Path donde se encuentra cv2
sys.path.append('/home/duckietown-t2/anaconda3/lib/python3.8/site-packages')
import cv2

#Path donde se encuentra las implementaciones de cyclegan
sys.path.append('/home/duckietown-t2/gym-duckietown/PyTorch-GAN/implementations/cyclegan')
from models import *
from datasets import *
from utils import *

import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

#valores por default
channels=3
img_height=256
img_width=256
input_shape = (channels, img_height, img_width)
n_residual_blocks=9
G_AB = GeneratorResNet(input_shape, n_residual_blocks)
G_BA = GeneratorResNet(input_shape, n_residual_blocks)

cuda = torch.cuda.is_available() #Para usar la tarjeta grafica dedicada

dataset_name='DuckieGan'
epoch=9 #numero del modelo

#Direccion en donde se encuentran los modelos
G_AB.load_state_dict(torch.load("/home/duckietown-t2/gym-duckietown/PyTorch-GAN/implementations/cyclegan/saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch),map_location='cpu'))

if cuda:
    G_AB = G_AB.cuda()
    
#transformación 
transforms_ = transforms.Compose([
    transforms.Resize((256, 256), 3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

archivo='/home/duckietown-t2/Downloads/probando6.mp4' #Ubicación del video

#Captura de los frames del video
cap = cv2.VideoCapture(archivo)

###Los comentarios con 3 '#' son para guardar el video simulado
###out=cv2.VideoWriter('/home/duckietown-t2/gym-duckietown/ola4.mp4',cv2.VideoWriter_fourcc(*'DIVX'),30,(640,480))
while (cap.isOpened()):
    
    ret, frame = cap.read() 
    
    if ret==False:
        break
    
    #array a imagen
    frame1=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    im = Image.fromarray(frame1)
    
    #aplica la tranformacion a la imagen
    imt = transforms_(im)
    imt = imt.unsqueeze(0)
    
    G_AB.eval()
    with torch.no_grad():
        if cuda: imt = imt.cuda()
        fake_B = G_AB(imt)

    #permutacion para numpy, y normalizacion de la imagen
    fake_B = fake_B[0,:,:,:].permute(1,2,0)
    fake_B = (fake_B * 0.5) + 0.5
    
    #Cambio de tamaño
    fake_B = cv2.resize(fake_B.cpu().numpy(), (640, 480))

    #Mas cosas raras
    img=cv2.cvtColor(fake_B, cv2.COLOR_RGB2BGR)*255
    img=img.astype(np.uint8)

    #Muestra el video original y su traducción
    cv2.imshow('video simulado',img)
    cv2.imshow('video',frame)
    
    ###out.write(img)
    #Tiempo entre cada frame y 'esc'para terminar
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
#Cierra el video    
cap.release()

#Guarda el video
###out.release()

