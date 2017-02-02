import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

def mask_change_Y_axis(mask,mask2):
    detect=np.sum(mask,axis=0)
    detect2=np.sum(mask2,axis=0)
    
    index=np.where(detect!=0)
    maskNew=np.array(mask)
    maskNew[:,index[0]]=mask2[:,index[0]]
    
    indexWith=np.where(detect2!=0)[0]
    indexWithout=np.where(detect2==0)[0]
    
    for i in indexWithout:
        if i in index[0]:
            indextocopy = np.argmin(np.abs(indexWith - i))
            maskNew[:,i]=maskNew[:,indexWith[indextocopy]]

    return maskNew
    
def mask_change_X_axis(mask,mask2):
    detect=np.sum(mask,axis=1)
    detect2=np.sum(mask2,axis=1)
    
    index=np.where(detect!=0)
    maskNew=np.array(mask)
    maskNew[index[0],:]=mask2[index[0],:]
    
    indexWith=np.where(detect2!=0)[0]
    indexWithout=np.where(detect2==0)[0]
    
    for j in indexWithout:
        if j in index[0]:
            indextocopy = np.argmin(np.abs(indexWith - j))
            maskNew[j,:]=maskNew[indexWith[indextocopy],:]
        
    return maskNew
    
    
def walls_Y_Detection(mask,mask2):
    UpWall=np.zeros(mask.shape)
    UpWall2=np.zeros(mask.shape)
    DownWall=np.zeros(mask.shape)
    DownWall2=np.zeros(mask.shape)
    yMax=mask.shape[0]
    
    maskup=mask[0:yMax-1,:]
    maskdown=mask[1:yMax,:]
    detect=maskup-maskdown
    whereDown=np.where(detect==1)
    DownWall[:,whereDown[1]] = np.tile(whereDown[0], (yMax,1))
    whereUp=np.where(detect==-1)
    UpWall[:,whereUp[1]] = np.tile(whereUp[0]+1, (yMax,1))


    maskup=mask2[0:yMax-1,:]
    maskdown=mask2[1:yMax,:]
    detect=maskup-maskdown
    whereDown=np.where(detect==1)
    DownWall2[:,whereDown[1]] = np.tile(whereDown[0], (yMax,1))
    whereUp=np.where(detect==-1)
    UpWall2[:,whereUp[1]] = np.tile(whereUp[0]+1, (yMax,1))
    
    return UpWall,UpWall2,DownWall,DownWall2
    
    
def walls_X_Detection(mask,mask2):
    #mask
    LeftWall=np.zeros(mask.shape)
    RightWall=np.zeros(mask.shape)
    LeftWall2=np.zeros(mask.shape)
    RightWall2=np.zeros(mask.shape)
    xMax=mask.shape[1]

    
    maskleft=mask[:,0:xMax-1]
    maskright=mask[:,1:xMax]
    detect=maskleft-maskright
    whereLeft=np.where(detect==-1)
    LeftWall[whereLeft[0],:] = np.tile(whereLeft[1]+1, (xMax,1)).transpose()
    whereRight=np.where(detect==1)
    RightWall[whereRight[0],:] = np.tile(whereRight[1], (xMax,1)).transpose()
    
    indexOpenR=np.where(mask[:,xMax-1]!=0)
    RightWall[indexOpenR[0],:]=xMax-1
    indexOpenL=np.where(mask[:,0]!=0)
    LeftWall[indexOpenL[0],:]=0


    maskleft=mask2[:,0:xMax-1]
    maskright=mask2[:,1:xMax]
    detect=maskleft-maskright
    whereLeft=np.where(detect==-1)
    LeftWall2[whereLeft[0],:]=np.tile(whereLeft[1]+1, (xMax,1)).transpose()
    whereRight=np.where(detect==1)
    RightWall2[whereRight[0],:]=np.tile(whereRight[1], (xMax,1)).transpose()
    
    indexOpenR2=np.where(mask2[:,xMax-1]!=0)
    RightWall2[indexOpenR2[0],:]=xMax-1
    indexOpenL2=np.where(mask2[:,0]!=0)
    LeftWall2[indexOpenL2[0],:]=0

    
    return LeftWall,LeftWall2,RightWall,RightWall2
    
def get_wall_displacement_X_axis(Wall1,Wall2):
    Gamma1= (Wall2 - Wall1)
    return Gamma1

def get_wall_displacement_Y_axis(Wall1,Wall2):
    Gamma1= (Wall2 - Wall1)
    return Gamma1


def get_distance_to_wall_right(X,Wall):
    XI1= (Wall-X)
    return XI1
    
def get_distance_to_wall_left(X,Wall):
    XI2= (X-Wall)
    return XI2
    
def get_distance_to_wall_up(Y,Wall):
    YI1= (Y-Wall)
    return YI1
    
def get_distance_to_wall_down(Y,Wall):
    YI2= (Wall-Y)
    return YI2
    
def get_translation_X_axis(X,XI1,XI2,Gamma1,Gamma2):
    Pi=(Gamma1)*(1-((XI1)/(XI2+XI1)))  +  (Gamma2)*(1-((XI2)/(XI2+XI1))) #Traslado
    return Pi
    
def get_translation_Y_axis(Y,YI1,YI2,Omega1,Omega2):
    Pj=(Omega1)*(1-((YI1)/(YI2+YI1)))  +  (Omega2)*(1-((YI2)/(YI2+YI1))) #Traslado
    return Pj
    
    
#TRASLADO ACOPLADO  
def coupled_mapping(X,Y,Pi,Pj,mask,field):
    y=Y[:,0]
    x=X[0,:]
    NewField=np.zeros(field.shape)
    
    for j in np.arange(Y.shape[0]):
        for i in np.arange(Y.shape[1]):
            if mask[j,i] != 0.:
                decI,entI=math.modf(Pi[j,i])#decimal part , integer part
                decI = np.abs(decI)
                newPosI = i+entI
                newPosNextI = newPosI + np.sign(Pi[j,i])*1
                
                decJ,entJ=math.modf(Pj[j,i])#decimal part , integer part
                decJ = np.abs(decJ)
                newPosJ = j+entJ
                newPosNextJ = newPosJ + np.sign(Pj[j,i])*1
                
                p00 = (1-decI)*(1-decJ)
                p10 = (1-decI) * (decJ) #arriba
                p11 = decI * decJ
                p01 = decI * (1-decJ) #derecha
    
                Ptot = p00+p10+p11+p01
                p00 = p00/Ptot
                p10 = p10/Ptot
                p11 = p11/Ptot
                p01 = p01/Ptot
                
                #print newPosI, newPosNextI, p00, p10 , p01 , p11
                
                NewField[newPosJ,newPosI]= NewField[newPosJ,newPosI] + field[j,i] * p00
                NewField[newPosJ,newPosNextI]= NewField[newPosJ,newPosNextI] + field[j,i] * p01
                NewField[newPosNextJ,newPosNextI]= NewField[newPosNextJ,newPosNextI] + field[j,i] * p11
                NewField[newPosNextJ,newPosI]= NewField[newPosNextJ,newPosI] + field[j,i] * p10     
    return NewField
  
def mapping_X_axis(X,Y,Pi,mask,field):
    y=Y[:,0]
    x=X[0,:]
    NewField=np.zeros(field.shape)
    for j in np.arange(Y.shape[0]):
        for i in np.arange(Y.shape[1]):
            if mask[j,i] != 0.:
                decI,entI=math.modf(Pi[j,i])#decimal part , integer part
                decI = np.abs(decI)
                newPosI = i+entI
                newPosNextI = newPosI + np.sign(Pi[j,i])*1
                
                p00 = (1-decI)
                p01 = decI #derecha
    
                Ptot = p00+p01
                p00 = p00/Ptot
                p01 = p01/Ptot
                

                
                NewField[j,newPosI]= NewField[j,newPosI] + field[j,i] * p00
                NewField[j,newPosNextI]= NewField[j,newPosNextI] + field[j,i] * p01
    
    return NewField


def mapping_Y_axis(X,Y,Pj,mask,field):
    y=Y[:,0]
    x=X[0,:]
    NewField=np.zeros(field.shape)
    for j in np.arange(Y.shape[0]):
        for i in np.arange(Y.shape[1]):
            if mask[j,i] != 0.:
                
                decJ,entJ=math.modf(Pj[j,i])#decimal part , integer part
                decJ = np.abs(decJ)
                newPosJ = j+entJ
                newPosNextJ = newPosJ + np.sign(Pj[j,i])*1
                
                p00 = (1-decJ)
                p10 = (decJ) #arriba
    
                Ptot = p00+p10
                p00 = p00/Ptot
                p10 = p10/Ptot
                
                NewField[newPosJ,i]= NewField[newPosJ,i] + field[j,i] * p00
                NewField[newPosNextJ,i]= NewField[newPosNextJ,i] + field[j,i] * p10
        
    return NewField
    
def smoothField_5points(field):
    fieldHalo=np.zeros((field.shape[0]+2,field.shape[1]+2))
    fieldHalo[1:field.shape[0]+1,1:field.shape[1]+1]=field
    xMax=fieldHalo.shape[1]
    yMax=fieldHalo.shape[0]
    
    maskup=fieldHalo[2:yMax,1:xMax-1]
    maskdown=fieldHalo[0:yMax-2,1:xMax-1]
    
    maskleft=fieldHalo[1:yMax-1,0:xMax-2]
    maskright=fieldHalo[1:yMax-1,2:xMax]
    
    maskSmooth = (fieldHalo[1:yMax-1,1:xMax-1] + maskup + maskdown + maskleft + maskright)/5
    #Think about make it conservative here. add the residual field of the extra halo rows and columns
    
    return maskSmooth
