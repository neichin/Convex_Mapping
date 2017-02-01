import netCDF4
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from Convex_Mapping_lib import *
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import matplotlib.pyplot as plt
import vtk 
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import griddata
from scipy.interpolate import interp2d



ncfile = netCDF4.Dataset('/Users/imerino/Documents/Convex_Mapping/melt_EXP1fiGt0040eene3f_COM_NEMO-CNRS.nc','a')
x= np.array(ncfile.variables['x'])[:] #240
y= np.array(ncfile.variables['y'])[:] #40
MeltRate = np.array(ncfile.variables['meltRate'])[0,:,:]
ncfile.close()

MeltRate = MeltRate * (MeltRate<1e10)

XMR, YMR = np.meshgrid(x,y)


reader = vtk.vtkXMLPUnstructuredGridReader()
path='/Users/imerino/Documents/These/MISMIP+/Occigen/Test500m_Schoof_SSAStar/Ice1r/ice1r0001.pvtu'
reader.SetFileName(path)
reader.Update()
output=reader.GetOutput()
Coords=vtk_to_numpy(output.GetPoints().GetData())
PointData=output.GetPointData()
numArrays=PointData.GetNumberOfArrays()
GL=vtk_to_numpy(PointData.GetArray(0))

infLimit=200000.
supLimit=480000
xG=np.arange(start=infLimit,stop=supLimit,step=500)
yG=np.arange(start=0,stop=80500,step=500)

X, Y = np.meshgrid(xG,yG)

indexX=np.where(Coords[:,0]>=infLimit)
index2=np.where(Coords[indexX[0],0]<=supLimit)

indexDef=indexX[0][index2]

GLGrid = griddata((Coords[indexDef,0],Coords[indexDef,1]), GL[indexDef], (X, Y), method='nearest')
MRInterpFunc = interp2d(x,y, MeltRate, kind='linear')

MRGrid = MRInterpFunc(xG,yG)

Mask1=1*(MRGrid!=0)

GLGrid = -1 * GLGrid *(GLGrid<0.)

plt.figure()
imgplot = plt.imshow(GLGrid,interpolation='nearest')


plt.figure()


imgplot = plt.imshow(Mask1,interpolation='nearest')

plt.show()



##########


maskExtraHalo=np.zeros((GLGrid.shape[0]+2,GLGrid.shape[1]+2))

mask2=np.array(maskExtraHalo)
mask2[1:GLGrid.shape[0]+1,1:GLGrid.shape[1]+1]=GLGrid

mask=np.array(maskExtraHalo)
mask[1:GLGrid.shape[0]+1,1:GLGrid.shape[1]+1]=Mask1

data=np.array(maskExtraHalo)
data[1:GLGrid.shape[0]+1,1:GLGrid.shape[1]+1]=MRGrid

XHalo=np.array(maskExtraHalo)
XHalo[1:GLGrid.shape[0]+1,1:GLGrid.shape[1]+1]=X

YHalo=np.array(maskExtraHalo)
YHalo[1:GLGrid.shape[0]+1,1:GLGrid.shape[1]+1]=Y

indices=np.indices(data.shape)


############


mask_X=mask_change_X_axis(mask,mask2) #component X de entre las dos maskaras


LeftWall,LeftWall2,RightWall,RightWall2 =  walls_X_Detection(mask,mask_X)
Gamma1 = get_wall_displacement_X_axis(LeftWall,LeftWall2)
Gamma2 = get_wall_displacement_X_axis(RightWall,RightWall2)
XI1=get_distance_to_wall_left(indices[1],LeftWall)
XI2=get_distance_to_wall_right(indices[1],RightWall)
Pi=get_translation_X_axis(XHalo,XI1,XI2,Gamma1,Gamma2)
NewIMG=mapping_X_axis(XHalo,YHalo,Pi,mask,data)


mask_Y=mask_change_Y_axis(mask_X,mask2)

UpWall,UpWall2,DownWall,DownWall2 = walls_Y_Detection(mask_X,mask_Y)

Omega1 = get_wall_displacement_Y_axis(UpWall,UpWall2)
Omega2 = get_wall_displacement_Y_axis(DownWall,DownWall2)

YI1=get_distance_to_wall_up(indices[0],UpWall)
YI2=get_distance_to_wall_down(indices[0],DownWall)

Pj=get_translation_Y_axis(YHalo,YI1,YI2,Omega1,Omega2)

NewIMG2=mapping_Y_axis(XHalo,YHalo,Pj,mask_X,NewIMG)



plt.figure()
imgplot = plt.imshow(data[:,:]*mask,interpolation='nearest')

plt.figure()
imgplot = plt.imshow(NewIMG2[:,:]*mask2,interpolation='nearest')

plt.show()