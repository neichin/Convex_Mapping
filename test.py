
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from Convex_Mapping_lib import *


img=np.array(mpimg.imread('/Users/imerino/Desktop/DriveLicense.png')[:,:,0])
#img=np.ones([8,8])


xMax=img.shape[1]
yMax=img.shape[0]

x=np.arange(img.shape[1])
y=np.arange(img.shape[0])

X,Y=np.meshgrid(x,y)

xMax=img.shape[1]
yMax=img.shape[0]



mask=np.ones(img.shape)

#mask[0:2,:]=0.
#mask[:,0:2]=0.
#mask[:,xMax-2:xMax]=0.
#mask[yMax-2:yMax,:]=0.

mask[0:50,:]=0.
mask[:,0:100]=0.
mask[:,xMax-50:xMax]=0.
mask[yMax-100:yMax,:]=0.

#mask2=np.array(mask)
#mask2[0:100,:]=0.
#mask2[:,0:100]=0.

mask2=np.ones(mask.shape)

mask2[0:100,:]=0.
mask2[yMax-100:yMax,:]=0.


mask2[:,0:20]=0.
mask2[:,xMax-20:xMax]=0.

#
#mask2[0:100,:]=0.
#mask2[yMax-100:yMax,:]=0.
#
#
#mask2[:,0:200]=0.
#mask2[:,xMax-100:xMax]=0.




mask_Y=mask_change_Y_axis(mask,mask2)

mask_X=mask_change_X_axis(mask,mask2)


LeftWall,LeftWall2,RightWall,RightWall2 =  walls_X_Detection(mask,mask_X)
Gamma1 = get_wall_displacement_X_axis(LeftWall,LeftWall2)
Gamma2 = get_wall_displacement_X_axis(RightWall,RightWall2)
XI1=get_distance_to_wall_left(X,LeftWall)
XI2=get_distance_to_wall_right(X,RightWall)
Pi=get_translation_X_axis(X,XI1,XI2,Gamma1,Gamma2)
NewIMG=mapping_X_axis(X,Y,Pi,mask,img)


mask_Y=mask_change_Y_axis(mask_X,mask2)

UpWall,UpWall2,DownWall,DownWall2 = walls_Y_Detection(mask_X,mask_Y)

Omega1 = get_wall_displacement_Y_axis(UpWall,UpWall2)
Omega2 = get_wall_displacement_Y_axis(DownWall,DownWall2)

YI1=get_distance_to_wall_up(Y,UpWall)
YI2=get_distance_to_wall_down(Y,DownWall)

Pj=get_translation_Y_axis(Y,YI1,YI2,Omega1,Omega2)

NewIMG2=mapping_Y_axis(X,Y,Pj,mask_X,NewIMG)


#LeftWall,LeftWall2,RightWall,RightWall2 =  walls_X_Detection(mask,mask_X)
#UpWall,UpWall2,DownWall,DownWall2 = walls_Y_Detection(mask,mask_Y)
#
#Gamma1 = get_wall_displacement_X_axis(LeftWall,LeftWall2)
#Omega1 = get_wall_displacement_Y_axis(UpWall,UpWall2)
#Gamma2 = get_wall_displacement_X_axis(RightWall,RightWall2)
#Omega2 = get_wall_displacement_Y_axis(DownWall,DownWall2)
#
#
#XI1=get_distance_to_wall_left(X,LeftWall)
#XI2=get_distance_to_wall_right(X,RightWall)
#YI1=get_distance_to_wall_up(Y,UpWall)
#YI2=get_distance_to_wall_down(Y,DownWall)
#
#Pi=get_translation_X_axis(X,XI1,XI2,Gamma1,Gamma2)
#Pj=get_translation_Y_axis(Y,YI1,YI2,Omega1,Omega2)
#
#NewIMG=mapping_X_axis(X,Y,Pi,mask,img)
#
#NewIMG2=mapping_Y_axis(X,Y,Pj,mask,NewIMG)

#NewIMG2=mapping_Y_axis(X,Y,Pj,mask,img)

plt.figure()
imgplot = plt.imshow(img[:,:]*mask,interpolation='nearest')

plt.figure()
imgplot = plt.imshow(NewIMG[:,:]*mask_X,interpolation='nearest')
#
#plt.figure()
#imgplot = plt.imshow(NewIMG2[:,:]*mask2,interpolation='nearest')

plt.show()


