"""
=================================
The amazing script for plotting EEG sensors on the scalp of two brains
=================================

"""
# Author: Guillaume Dumas <guillaume.dumas@pasteur.fr>
# Author: Amir Djalovski <Amir.Djv@gmail.com>
# License: BSD (3-clause)

import sys, os

path = "D:/Amir/Dropbox/Studies BIU/Ruth Feldman/My Thesis/My Analysis/EEG/EEG/shared/hypyviz"
os.chdir(path)

#Load support functions
exec(open("src/support_functions.py").read())

#Loading data files
import mne
import numpy as np

epochsS1 = mne.read_epochs("data/subject1.fif", preload=True)
epochsS2 = mne.read_epochs("data/subject2.fif", preload=True)
combined = combineEpochs(epochsS1 = epochsS1, epochsS2 = epochsS2)

print(combined.info["ch_names"])

#Calculating locations
from mpl_toolkits.mplot3d import Axes3D
from copy import copy
import numpy as np

locations = copy(np.array([ch['loc'] for ch in combined.info['chs']]))
cap1_locations = locations[:31, :3]
print("Mean: ", np.nanmean(cap1_locations, axis=0))
print("Min: ", np.nanmin(cap1_locations, axis=0))
print("Max: ", np.nanmax(cap1_locations, axis=0))

translate = [0, 0.25, 0]
rotZ = np.pi

cap2_locations = copy(cap1_locations)
newX = cap2_locations[:, 0] * np.cos(rotZ) - cap2_locations[:, 1] * np.sin(rotZ)
newY = cap2_locations[:, 0] * np.sin(rotZ) + cap2_locations[:, 1] * np.cos(rotZ)
cap2_locations[:, 0] = newX
cap2_locations[:, 1] = newY
cap2_locations = cap2_locations + translate
print("Mean: ", np.nanmean(cap2_locations, axis=0))
print("Min: ", np.nanmin(cap2_locations, axis=0))
print("Max: ", np.nanmax(cap2_locations, axis=0))
sens_loc = np.concatenate((cap1_locations, cap2_locations), axis=0)

#testing that the new locations and the old locations are at the same length
assert len([ch['loc'] for ch in combined.info['chs']]) == len(sens_loc), "the caps locations are not in the same length"

#Changing location 
for old, new in enumerate(sens_loc):
    combined.info["chs"][old]["loc"][0:3] = new[0:3]

locationSettings = combined.info["chs"].copy()

del cap1_locations, cap2_locations, old, new, newX, newY, rotZ, translate

###############################################################################
#############################Start of viz######################################
###############################################################################

########################Temp Idea for curves##########################

import numpy as np
from scipy import interpolate
import mayavi.mlab as mlab
from scipy import linalg, stats


#Plot EEG cap with electrode names 
fig = mlab.figure(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))
points = mlab.points3d(sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2],
                  color=(1, 1, 1), opacity=1, scale_factor=0.005,
                  figure=fig)

nodes_shown = list(range(0, 62))

chNames = []
#Changing channels name as letters -M / -F
for i, c in enumerate(combined.info["ch_names"]):
    if c[-1] == "0":
        chNames.append(c[:-1] + "M")
    elif c[-1] == "1":
        chNames.append(c[:-1] + "F")

#Channels name as letters -M / -F
picks = np.array(list(range(0, len(chNames))))

for node in nodes_shown:
    x, y, z = sens_loc[node]
    mlab.text3d(x, y, z, chNames[picks[node]],
                scale=0.005,
                color=(0, 0, 0))


#Start of ploting the lines between electrodes. 
#I cannot show the lines on the same plot. 

#This will probably be in loop 
###inputs

p1 = sens_loc[0] #starting point 
p2 = sens_loc[35] #end point
npts = 100 # number of points to sample between the start/end

#creating data to connect the dots 
y = np.array([0,.5,.75,.75,.5,0]) #describe shape in 1d 
amp = 5 #curve height factor. bigger means heigher 

#get the adder. This will be used to raise the z coords
x = np.arange(y.size)
xnew = np.linspace(x[0],x[-1] , npts) #sample the x coord
tck = interpolate.splrep(x,y,s=0) 
adder = interpolate.splev(xnew,tck,der=0)*amp
adder[0] = adder[-1] = 0
adder = adder.reshape((-1,1))

#get a line between points
shape3 = np.vstack([np.linspace(p1[dim],p2[dim],npts) for dim in range(3)]).T

#raise the z coordinate
shape3[:,-1] = shape3[:,-1] + adder[:,-1]

#plot
x, y, z = (shape3[:,dim] for dim in range(3))
mlab.points3d(x, y, z, color=(0,0,0))
mlab.plot3d(x,y,z,tube_radius=0.1)





####parts to ignore / scramble attempts / mess / bits and pieces

for xi, yi, zi in zip(x, y, z):
    lines = mlab.plot3d([xi, xi], [yi, yi], [zi, zi], [val, val],
                             vmin=vmin, vmax=vmax, tube_radius=0.2, #color features
                             colormap='blue-red')
    lines.module_manager.scalar_lut_manager.reverse_lut = True


#For straight lines
vmax = np.max(con_val)
vmin = np.min(con_val)
for val, nodes in zip(con_val, con_nodes):
    x1, y1, z1 = sens_loc[nodes[0]]
    x2, y2, z2 = sens_loc[nodes[1]]
    lines = mlab.plot3d([x1, x2], [y1, y2], [z1, z2], [val, val],
                             vmin=vmin, vmax=vmax, tube_radius=0.0002, #color features
                             colormap='blue-red')
    lines.module_manager.scalar_lut_manager.reverse_lut = True



##Calculate connectivity for single cap
from mne.channels import find_ch_connectivity

x, y = find_ch_connectivity(epochsS1.info, ch_type="eeg")

from scipy.sparse import csr_matrix
A = csr_matrix(x.toarray())

##Plot two caps connectivity
def capTest(x):
    if x >= 31:
        out = x - 31
    else:
        out = x 
    return(out)

#Brain areas
centerS1  = ['Cz-0', 'Fz-0', 'Pz-0']
leftTemporalS1 = ["FT9-0", "TP9-0", "T7-0"]
rightTemporalS1 = ["FT10-0", "TP10-0",  "T8-0"]

chToTake = [i[:-2] for i in centerS1 + leftTemporalS1 + rightTemporalS1]
#Define conections
con = np.zeros([62, 62])
for e1 in range(62):
    for e2 in range(62):
        if combined.info["ch_names"][e1][:-2] in chToTake and combined.info["ch_names"][e2][:-2] in chToTake:
            k1, k2 = list(map(capTest, [e1, e2]))
            if A[k1, k2] and e1 <= 30 and e2 >= 31:
                con[e1][e2] = 1

import matplotlib.pyplot as plt
plt.spy(con)




#######
# Get the strongest connections
n_con = len(con)**2  # show up to 100 connections
min_dist = 0.01  # exclude sensors that are less than 5cm apart
threshold = np.sort(con, axis=None)[-n_con] #sort the con by size and pick the index of n_con
ii, jj = np.where(con > 0)

# Remove close connections
con_nodes = list()
con_val = list()
for i, j in zip(ii, jj):
    if linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
        con_nodes.append((i, j))
        con_val.append(con[i, j])

con_val = np.array(con_val)

# Show the connections as tubes between sensors
#By General - all in the same color.
vmax = np.max(con_val)
vmin = np.min(con_val)
for val, nodes in zip(con_val, con_nodes):
    x1, y1, z1 = sens_loc[nodes[0]]
    x2, y2, z2 = sens_loc[nodes[1]]
    lines = mlab.plot3d([x1, x2], [y1, y2], [z1, z2], [val, val],
                             vmin=vmin, vmax=vmax, tube_radius=0.0002,
                             colormap='blue-red')
    lines.module_manager.scalar_lut_manager.reverse_lut = True






import mne
from mne.viz import plot_alignment

print(__doc__)


#Template#
data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'
trans = mne.read_trans(data_path + '/MEG/sample/sample_audvis_raw-trans.fif')
raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')

fig = plot_alignment(raw.info, trans, subject='sample', dig=False,
                     eeg=['original', 'projected'], meg=[],
                     coord_frame='head', subjects_dir=subjects_dir)

fig = plot_alignment(combined.info, trans, subject='sample', dig=False,
                     eeg=['original', 'projected'], meg=[],
                     coord_frame='head', subjects_dir=subjects_dir)

fig = plot_alignment(combined.info,
                     trans = None,
                     subject='sample',
                     dig=False,
                     eeg=['projected'],
                     meg=False,
                     coord_frame='head',
                     subjects_dir=subjects_dir)
