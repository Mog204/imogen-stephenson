# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:09:39 2023

@author: imy1
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#from sweep_final_final import getmagneticfield_side_Yfit
#from sweep_final_final import getmagneticfield_side_Xfit
import pickle as pkl

fsize=20

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fsize
plt.rcParams.update({'mathtext.default': 'it',
                    'mathtext.fontset': "dejavuserif"})
csfont = {'fontname':'Times New Roman'}

#GET SIMULATED DATA

mu_0 = 4 *np.pi *1e-7

#Coil parameters
turns = 100
#length = 0.02 #in metres
current = 1 #in AMperes
#Magnetic field within the coil is therefore:
mu_0 = 4*np.pi*1e-7
radius = 0.05 #in metres
CSA = np.pi*radius**2
z0 = 0.325
x0 = 0.13
xf = 0.09
step = 0.01

nsteps = 5



y = z0






magmoment_coil = current* turns * CSA

#2D SIMULATION
def field_from_mag_dipole(r,m):
    rn = np.sqrt(r[0]**2+r[1]**2)
    return mu_0/(4*np.pi) * ((3 * np.dot(r, np.dot(m,r))/(rn**5)) - m/(rn**3))


r=np.zeros((nsteps,2))
rmag = np.zeros((nsteps))
for i in range(nsteps):
    r[i,:] = np.array([x0 - (step*i), z0])
    rmag[i] = np.sqrt(r[i,0]**2 + r[i,1]**2)

m_mag = turns*current*np.pi*radius**2
m_dir = np.array([0,1])


Bvec = np.zeros((nsteps,2))
Bmag = np.zeros((nsteps))
for i in range(nsteps):
    Bvec[i,:] = field_from_mag_dipole(r[i,:],m_mag*m_dir)
    Bmag[i] = np.sqrt(Bvec[i,0]**2 + Bvec[i,1]**2)

print(Bvec)
print('\n')
print(Bmag)

    
print('THESE USE THE SCALAR FORMULA')

Bmag_test = np.zeros((nsteps))
for i in range(nsteps):
    Bmag_test[i] = (mu_0*m_mag * np.cos(np.arctan(r[i,0]/r[i,1])))/(2*np.pi*rmag[i]**3)
    
print(Bmag_test)

print('Ratio')
print(Bmag_test/Bmag)

#This is for the simulation






def zerofunc(variables, B1,y, magmoment_coil):
    theta1 = variables  # Assign names to the variables for clarity
    # Define your system of equations here
    #eq1 = y*(((2*np.pi*B1)/(mu_0*magmoment_coil))**(1/3)) - (np.cos(theta1)*(np.cos(theta1)**(1/3)))
    eq1 = y/(np.cos(theta1)) - ((mu_0*magmoment_coil/(4*np.pi*B1))**(1/3) * (4*np.cos(theta1)**2 + np.sin(theta1)**2)**(1/6))
    return np.sqrt(eq1**2)



def coordinates_1D_singlesource(B1, y, radius=radius, magmoment_coil = magmoment_coil ):
    rxry = []
    for i, B_i in enumerate(B1):
        B1_i = B1[i]
        initial_guess = np.pi/4
        args = (B1_i, y, magmoment_coil)

        results1 = sp.optimize.minimize(zerofunc,
                              initial_guess, 
                              args=args,
                              method="SLSQP",
                              bounds=[
                                  (0, np.pi/2)
                                  ]
                              )


        #theta1_fitted = np.arccos((2*np.pi)**(1/4) * (y * (B1_i/(magmoment_coil*mu_0))**(1/3))**(3/4))
        
        theta1_fitted = results1.x[0]
        r1_fitted = y/np.cos(theta1_fitted)
        r1_x = r1_fitted*np.sin(theta1_fitted)
        r1_y = r1_fitted*np.cos(theta1_fitted)
        print("FUNCTION OUTPUT, NEW RUN \n")
        print('\n For B value after step '+str(i)+': /n"')
        print('r1 : ',r1_fitted)
        print('theta1 : ',theta1_fitted)
        print('r1_x : ', r1_x)
        print('r1_y : ', r1_y)
        
        print('This should be '+str(y)+' : ', r1_fitted * np.cos(theta1_fitted))
        rxry.append(r1_x)
    return np.array(rxry)

#SIMULATION TEST 

Bsim_info = coordinates_1D_singlesource(Bmag, y)
print('This should be ', r[:], ' : \n', Bsim_info[:])


fig, ax = plt.subplots(1,1)
ax.plot([0,1,2,3,4], Bsim_info[:]*100, c="k",label="Computed displacement")
ax.plot([0,1,2,3,4], r[:,0]*100, c="b", label="Real displacement")
# Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_ylabel('Displacement (cm)')
ax.set_xlabel('Steps')
# Put a legend to the right of the current axis
ax.legend(bbox_to_anchor=(1,1))
#fig.tight_layout()
fig.savefig('distance_movements_sim.pdf')











#NOW USING ACTUAL EXPERIMENTAL DATA
#This is for the simulation

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle as pkl

# =============================================================================
# 
# if False:
#     abs_Blab_0to4cm_777 = getmagneticfield_side_Yfit(777,5, 5,100,100,1e-7,0.1e-7)
#     with open('777.pkl', 'wb') as k:
#         pkl.dump(abs_Blab_0to4cm_777,k)
# else:
# =============================================================================
with open('777.pkl','rb') as k:
        abs_Blab_0to4cm_777=pkl.load(k)
        
        
# =============================================================================
# if False:
#     abs_Blab_0to4cm_976 = getmagneticfield_side_Xfit(976,5, 5,100,100,1e-7,0.1e-7)
#     with open('976.pkl', 'wb') as k:
#         pkl.dump(abs_Blab_0to4cm_976,k)
# else:
# =============================================================================
with open('976.pkl','rb') as k:
        abs_Blab_0to4cm_976=pkl.load(k)
#import the data

print(abs_Blab_0to4cm_777)
print(abs_Blab_0to4cm_976)

#was 13 to the right of first source,  7 to the left of the other
#was 35cm down from both

# =============================================================================
# r777_0cm = np.sqrt(0.07**2 + 0.325**2)
# r976_0cm = np.sqrt(0.13**2 + 0.325**2)
# theta777_0cm = np.arctan(0.325/0.07)
# theta976_0cm = np.arctan(0.325/0.13)
# =============================================================================

r777_0cm = np.sqrt(0.13**2 + 0.325**2)
r976_0cm = np.sqrt(0.07**2 + 0.325**2)
theta777_0cm = np.arctan(0.13/0.325)
theta976_0cm = np.arctan(0.07/0.325)

r777_1cm = np.sqrt(0.12**2 + 0.325**2)
r976_1cm = np.sqrt(0.08**2 + 0.325**2)
theta777_1cm = np.arctan(0.12/0.325)
theta976_1cm = np.arctan(0.08/0.325)

#Coil parameters
turns = 100
#length = 0.02 #in metres
current = 1 #in AMperes
#Magnetic field within the coil is therefore:
mu_0 = 4*np.pi*1e-7
radius = 0.05 #in metres
CSA = np.pi*radius**2



d = 0.2 #distance_between_coils is 20cm

y = z0 #distance from top

B777= abs_Blab_0to4cm_777[1]*1e-4
B976 = abs_Blab_0to4cm_976[1]*1e-4
B777_0cm, B777_1cm, B777_2cm, B777_3cm = B777
B976_0cm, B976_1cm, B976_2cm, B976_3cm = B976 #they were in Gauss, convert to T.

#Solve for turns.current of the experiment given a known location
M777 = 4*np.pi*(r777_0cm**3)*B777_0cm/(mu_0*(4*np.cos(theta777_0cm)**2 + np.sin(theta777_0cm)**2)**(1/2))
M976 = 4*np.pi*(r976_0cm**3)*B976_0cm/(mu_0*(4*np.cos(theta976_0cm)**2 + np.sin(theta976_0cm)**2)**(1/2))
#see if they are the same:
print("777 M : ", M777, "\n 976 M: ", M976)

Mav = np.mean([M777, M976])
#Mav= M777
#Mav = M976


magmoment_coil = Mav


#EXPERIMENT TEST 

Bexp_info_777 = coordinates_1D_singlesource(B777, y, magmoment_coil = Mav)
print('This should be ', [0.13, 0.12, 0.11, 0.10], ' : \n', Bexp_info_777[:])


Bexp_info_976 = coordinates_1D_singlesource(B976, y, magmoment_coil = Mav)
print('This should be ', [0.07, 0.08, 0.09, 0.10], ' : \n', Bexp_info_976[:])


print('This should be around 0.2 for 1D single source to work : ', Bexp_info_777[:]+Bexp_info_976[:])  

fig, ax = plt.subplots(1,1)
ax.plot([0,1,2,3], Bexp_info_777[:]*100, c="deeppink",label="777 Hz")
ax.plot([0,1,2,3], Bexp_info_976[:]*100, c="indigo", label="976 Hz")
# Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_ylabel('Displacement (cm)')
ax.set_xlabel('Increments')
# Put a legend to the right of the current axis
ax.legend(loc="center right")
#fig.tight_layout()
fig.tight_layout()
fig.savefig('distance_movements_exp.pdf')



####SEPARATING OUT MAGNETIC DIPOLE MOMENT TO COMPENSATE FOR THE DIFFERENCE BETWEEN THEM
M976_1cm = 4*np.pi*(r976_1cm**3)*B976_1cm/(mu_0*(4*np.cos(theta976_1cm)**2 + np.sin(theta976_1cm)**2)**(1/2))

B777_only = coordinates_1D_singlesource(B777, y, magmoment_coil = M777)
print('This should be ', [0.13, 0.12, 0.11, 0.10], ' : \n', B777_only[:])

B976_only = coordinates_1D_singlesource(B976, y, magmoment_coil = M976)
print('This should be ', [0.07, 0.08, 0.09, 0.10], ' : \n', B976_only[:])


fig, ax = plt.subplots(1,1)
ax.plot([0,1,2,3], B777_only[:]*100, c="deeppink",label="777 Hz")
ax.plot([0,1,2,3], B976_only[:]*100, c="indigo", label="976 Hz")
# Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_ylabel('Displacement (cm)')
ax.set_xlabel('Increments')
# Put a legend to the right of the current axis
ax.legend(bbox_to_anchor=(1,1))
#fig.tight_layout()
fig.tight_layout()
fig.savefig('distance_movements_exp_differentM.pdf')


fig, ax = plt.subplots(1,1)
ax.plot([0,1,2,3], (B777_only[:]-B777_only[0])*100, c="deeppink",label="777 Hz")
ax.plot([0,1,2,3], (B976_only[:]-B976_only[0])*100, c="indigo", label="976 Hz")
# Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_ylabel('Displacement (cm)')
ax.set_xlabel('Increments')
# Put a legend to the right of the current axis
ax.legend(loc="lower left")
#fig.tight_layout()
fig.tight_layout()
fig.savefig('distance_movements_exp_differentM_diff.pdf')



"""

















#====================== TWO SOURCE METHOD ============

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle as pkl


# =============================================================================
# if False:
#     abs_Blab_0to4cm_777 = getmagneticfield_side_Yfit(777,5, 5,100,100,1e-7,0.1e-7)
#     with open('777.pkl', 'wb') as k:
#         pkl.dump(abs_Blab_0to4cm_777,k)
# else:
# =============================================================================
with open('777.pkl','rb') as k:
        abs_Blab_0to4cm_777=pkl.load(k)
        
        
# =============================================================================
# if False:
#     abs_Blab_0to4cm_976 = getmagneticfield_side_Xfit(976,5, 5,100,100,1e-7,0.1e-7)
#     with open('976.pkl', 'wb') as k:
#         pkl.dump(abs_Blab_0to4cm_976,k)
# else:
# =============================================================================
with open('976.pkl','rb') as k:
        abs_Blab_0to4cm_976=pkl.load(k)
#import the data

print(abs_Blab_0to4cm_777)
print(abs_Blab_0to4cm_976)

#was 13 to the right of first source,  7 to the left of the other
#was 35cm down from both

# =============================================================================
# r777_0cm = np.sqrt(0.07**2 + 0.325**2)
# r976_0cm = np.sqrt(0.13**2 + 0.325**2)
# theta777_0cm = np.arctan(0.325/0.07)
# theta976_0cm = np.arctan(0.325/0.13)
# =============================================================================

r777_0cm = np.sqrt(0.13**2 + 0.325**2)
r976_0cm = np.sqrt(0.07**2 + 0.325**2)
theta777_0cm = np.arctan(0.13/0.325)
theta976_0cm = np.arctan(0.07/0.325)

r777_1cm = np.sqrt(0.12**2 + 0.325**2)
r976_1cm = np.sqrt(0.08**2 + 0.325**2)
theta777_1cm = np.arctan(0.12/0.325)
theta976_1cm = np.arctan(0.08/0.325)

#Coil parameters
turns = 100
#length = 0.02 #in metres
current = 1 #in AMperes
#Magnetic field within the coil is therefore:
mu_0 = 4*np.pi*1e-7
radius = 0.05 #in metres
CSA = np.pi*radius**2



d = 0.2 #distance_between_coils is 20cm

y = z0 #distance from top

B777= abs_Blab_0to4cm_777[1]*1e-4
B976 = abs_Blab_0to4cm_976[1]*1e-4
B777_0cm, B777_1cm, B777_2cm, B777_3cm = B777
B976_0cm, B976_1cm, B976_2cm, B976_3cm = B976 #they were in Gauss, convert to T.

#Solve for turns.current of the experiment given a known location
M777 = 4*np.pi*(r777_0cm**3)*B777_0cm/(mu_0*(4*np.cos(theta777_0cm)**2 + np.sin(theta777_0cm)**2)**(1/2))
M976 = 4*np.pi*(r976_0cm**3)*B976_0cm/(mu_0*(4*np.cos(theta976_0cm)**2 + np.sin(theta976_0cm)**2)**(1/2))
#see if they are the same:
print("777 M : ", M777, "\n 976 M: ", M976)

Mav = np.mean([M777, M976])
#Mav= M777
#Mav = M976


def zerofunc_2source(variables, B1,B2, d, m1, m2):
    r1, theta1, r2, theta2= variables  # Assign names to the variables for clarity
    dx, dy, dz = d
    # Define your system of equations here
    eq1 = ((mu_0*m1/(4*np.pi*B1))**(1/3) * (4*np.cos(theta1)**2 + np.sin(theta1)**2)**(1/6)) - (dx + (r2 * np.sin(theta2))/np.sin(theta1)) + ((mu_0*m2/(4*np.pi*B2))**(1/3) * (4*np.cos(theta2)**2 + np.sin(theta2)**2)**(1/6)) - (dz + (r2 * np.cos(theta1))/np.cos(theta2))
    return np.sqrt(eq1**2)

magmoment_coil = Mav

def coordinates_2D_twosource(B1, B2, d, m1, m2,radius=radius ):
    rx1rx2 = []
    for i, B_i in enumerate(B1):
        B1_i = B1[i]
        B2_i = B2[i]
        initial_guess = [r777_0cm, theta777_0cm, r976_0cm, theta976_0cm]
        args = (B1_i, B2_i, d, m1, m2)

        results1 = sp.optimize.minimize(zerofunc_2source,
                              initial_guess, 
                              args=args,
                              method="SLSQP",
                              bounds=[
                                  (0, np.pi/2)
                                  ]
                              )


        #theta1_fitted = np.arccos((2*np.pi)**(1/4) * (y * (B1_i/(magmoment_coil*mu_0))**(1/3))**(3/4))
        
        r1_fitted, theta1_fitted, r2_fitted, theta2_fitted = results1.x
        r1_x = r1_fitted*np.sin(theta1_fitted)
        r1_y = r1_fitted*np.cos(theta1_fitted)
        r2_x = r2_fitted*np.sin(theta2_fitted)
        r2_y = r2_fitted*np.cos(theta2_fitted)
        print("FUNCTION OUTPUT, NEW RUN \n")
        print('\n For B value after step '+str(i)+': /n"')
        print('r1 : ',r1_fitted)
        print('theta1 : ',theta1_fitted)
        print('r2 : ',r2_fitted)
        print('theta2 : ',theta2_fitted)
        print('r1_x : ', r1_x)
        print('r2_y : ', r2_y)
        
        print('This should be '+str(y)+' : ', r1_y, r2_y)
        rx1rx2.append([r1_x, r2_x])
    return np.array(rx1rx2)

d = [0.2, 0, 0]
values_2source = coordinates_2D_twosource(B777, B976, d, M777, M976)
print('This should be ', [0.13, 0.12, 0.11, 0.10], ' : \n', values_2source[:,0])
print('This should be ', [0.07, 0.08, 0.09, 0.10], ' : \n', values_2source[:,1])


fig, ax = plt.subplots(1,1)
ax.plot([0,1,2,3], values_2source[:,0]*100, c="k",label="Displacement from 777 source")
ax.plot([0,1,2,3], values_2source[:,1]*100, c="b", label="Displacement from 976 source")
# Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_ylabel('Calculated displacement (cm)')
ax.set_xlabel('Steps')
# Put a legend to the right of the current axis
ax.legend(loc='upper right')
#fig.tight_layout()
fig.savefig('2source_distance_movements_exp_differentM.pdf')

"""