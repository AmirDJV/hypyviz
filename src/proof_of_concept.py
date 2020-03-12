# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:27:46 2020

@author: Amir
"""

import numpy as np
from scipy import interpolate
from mayavi import mlab
import sys 

###inputs

p1 = np.array([0,0,0]) #first point
p2 = np.array([1,1,1])  #second point

p1 = np.array([-0.0294367,  0.0839171, -0.00699]) #first point
p2 = np.array([0.0502438, 0.1968888, 0.042192])  #second point

#p1=np.random.uniform(0,20,(3)) #first point
#p2=np.random.uniform(0,20,(3)) #second point

npts = 100 # number of points to sample
y=np.array([0,.5,.75,.75,.5,0]) #describe your shape in 1d like this
amp=5 #curve height factor. bigger means heigher 

#get the adder. This will be used to raise the z coords
x=np.arange(y.size)
xnew = np.linspace(x[0],x[-1] , npts) #sample the x coord
tck = interpolate.splrep(x,y,s=0) 
adder = interpolate.splev(xnew,tck,der=0)*amp
adder[0]=adder[-1]=0
adder=adder.reshape((-1,1))

#get a line between points
shape3=np.vstack([np.linspace(p1[dim],p2[dim],npts) for dim in range(3)]).T

#raise the z coordinate
shape3[:,-1]=shape3[:,-1]+adder[:,-1]

#plot
x,y,z=(shape3[:,dim] for dim in range(3))
mlab.points3d(x,y,z,color=(0,0,0))
mlab.plot3d(x,y,z,tube_radius=0.2)
mlab.text3d(p1[0],p1[1],p1[2],"point1",
                scale=0.5,
                color=(0, 0, 0))
mlab.text3d(p2[0],p2[1],p2[2],"point2",
                scale=0.5,
                color=(0, 0, 0))

mlab.outline()
mlab.axes()
mlab.show()