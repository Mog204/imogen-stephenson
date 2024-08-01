# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:09:36 2023

@author: imy1
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import minimize
import lvm_read
import h5py
from scipy.spatial.transform import Rotation as R

fsize=22

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fsize
#plt.rcParams.update({'mathtext.default': 'it',
#                    'mathtext.fontset': "dejavuserif"})
#csfont = {'fontname':'Arial'}

####DEFINING HAMILTONIAN FUNCTION
from scipy.spatial.transform import Rotation as R

def R_theta_about_u(theta,u):
    u = u/np.linalg.norm(u)
    u *= theta
    return R.from_rotvec(u)



#For reverse z
R111_family = [R_theta_about_u(np.arctan(np.sqrt(2)),[1,-1,0]), #For [111] to x axis
                        R_theta_about_u(-np.pi/2-np.arctan(1/np.sqrt(2)),[1,1,0]), #[1,-1,-1]
                        R_theta_about_u(-np.arctan(np.sqrt(2)),[1,-1,0]), # [-1,-1,1]
                        R_theta_about_u(np.pi/2+np.arctan(1/np.sqrt(2)),[1,1,0])]   #[-1,1,-1]


#Define B_lab (magnetic field in frame of lab)
B_lab=np.array([2,5,6]) 
def rot(n,B_lab):
    rotated_B_values=[]
    for i in range(n):
        Ri=R111_family[i]
        B=Ri.apply(B_lab)
        rotated_B_values.append(B)
        #print(rotated_B_values)
    final_B=[]
    for i,B in enumerate(rotated_B_values):
        final_B.append([i+1,B])
    return final_B

def nv_components(B_lab):        
    B=rot(4,B_lab) 
    #print("B =",B)
    
    B_nv=[]
    for i in range(len(B)):
        B_nv.append(B[i][1])
    #print("B_nv = ", B_nv)
    
    B_nv_x =[]
    for i in range(len(B_nv)):
        B_nv_x.append(B_nv[i][0])
        
    B_nv_y =[]
    for i in range(len(B_nv)):
        B_nv_y.append(B_nv[i][1])
    
    B_nv_z =[]
    for i in range(len(B_nv)):
        B_nv_z.append(B_nv[i][2])
    """    
    #checking that magnitude stays the same:
    for i in range(len(B_nv)):
        tol = 1.e-6
        print(np.linalg.norm(B_nv[i]),", ", np.linalg.norm(B_lab),", ", np.linalg.norm(B_nv[i])-np.linalg.norm(B_lab)<tol)
    """
    return B_nv, B_nv_x,B_nv_y,B_nv_z
    

B_nv,B_nv_x,B_nv_y,B_nv_z = nv_components(B_lab)



#DEFINING HAMILTONIAN

Sx=(1/np.sqrt(2))*np.array([[0,1,0],[1,0,1],[0,1,0]])
Sy=(1/(np.sqrt(2)*1j))*np.array([[0,1,0],[-1,0,1],[0,-1,0]])
Sz=np.array([[1,0,0],[0,0,0],[0,0,-1]])
IM=np.array([[1,0,0],[0,1,0],[0,0,1]])
Sp = (Sx + 1.j*Sy)/np.sqrt(2) # i added this sqrt(2) here .. I think it makes more sense
Sm = (Sx - 1.j*Sy)/np.sqrt(2)
S=np.array([Sx,Sy,Sz])

Sh = (Sp+Sm)/np.sqrt(2)

def Hfuncsingle(D, B=[0,0,0]):
    h_red_eV = 6.582116e-16
    g = 2
    mu_B = 5.7884e-11
    #l = g*mu_B/(h_red_eV*2*np.pi)
    l=2.799
    EF=5
    #D=2870
    #D=2740
    #Hamil=D*(np.dot(Sz,Sz)- (2/3)*IM) + (l*(Bx*Sx + By*Sy + Bz*Sz)) #+ EF*(np.dot(Sx.T,Sx) - np.dot(Sy.T,Sy))
    Hamil=D*(np.dot(Sz,Sz)- (2/3)*IM) + (l*(B[0]*Sx + B[1]*Sy + B[2]*Sz)) #+ EF*(np.dot(Sx.T,Sx) - np.dot(Sy.T,Sy))
    return Hamil
    

def Hamiltonian(B_x,B_y,B_z,n,D):

    vals=[]
    vecs=[]
    for k in range(n):
        Bx=B_x[k]
        By=B_y[k]
        Bz=B_z[k]
        
        val,vec=np.linalg.eig(Hfuncsingle(D=D, B=[Bx,By,Bz])) #This gives eigenvalues and eigenvectors in tuple
        vals.append(val)
        vecs.append(vec)
    return np.array(vals),np.array(vecs)
    
   
    
#finding which eigenvalues correspond to 0 energy (trivial solutions)
#For each NV centre axis, find the "0" state of it (the one that primarily has y component and not others) so we can consider transitions from this state to the others. 
#We will remove these "0" state values so we don't treat them as excitation ones. 
def findzero(eigenvectors):
    zero_state=[0,1,0]
    zero_vals=[]
    index_to_remove=[]
    zero_int=[]
    for i,vec in enumerate(eigenvectors):
        v=eigenvectors[i]
        v_zero_like=[]
        for ind, vv in enumerate(v):
            zero_int=np.abs(np.dot(np.conjugate(zero_state),v[ind]))**2
            v_zero_like.append(zero_int)
            #print("zero state values, ",ind," : ",zero_int)
        #r=v_int.index(np.max(v_int))
        zero_index=np.argmax(v_zero_like)
        index_to_remove.append(zero_index)
        zero_value=v[zero_index]        
    return np.array(index_to_remove)

#print(findzero(Hamiltonian(B_nv_x,B_nv_y,B_nv_z,4)[1]))


#Find the transition E values for +1/-1
def res_values(vals, vecs,index_to_remove, n):
    #vals,vecs = Hamiltonian(B_x,B_y,B_z,n)
    #zero_vals,index_to_remove = findzero(vecs)
    res_values=[]
    for i in range(len(vals)):
        eigenvals=vals[i]
        new_eigenvals=np.delete(eigenvals, index_to_remove[i]) #gets rid of the reference so we aren't left with 0s
        res_diffs_i=new_eigenvals - eigenvals[index_to_remove[i]]
        res_values.append(res_diffs_i)
    return np.abs(res_values)

#This outputs the energy resonances for transitions from the 0 state (they are the energy shifts associated with transition due to the magnetic field)

def resonances_fermi(B_lab,D=2870, operator=Sh):
    #B_lab = np.array([Bx,By,Bz])
    n=4
    B_nv,B_nv_x,B_nv_y,B_nv_z = nv_components(B_lab)
    vals,vecs = Hamiltonian(B_nv_x,B_nv_y,B_nv_z,n, D=D)
    index_to_remove = findzero(vecs)
    energies_MHz = res_values(vals,vecs,index_to_remove,n)
    transition_elements=[]
    spin_states=[]
    for i,vec in enumerate(vecs):
        transition_element=(np.abs(np.dot(np.conjugate(vecs[i]), np.dot(operator, vec[index_to_remove[i]]))) ** 2)
        transition_element=np.delete(transition_element,index_to_remove[i])
        transition_elements.append(transition_element)
        spin_states.append(np.array([1,-1]))
            
    transition_elements=np.array(transition_elements)
    spin_states = np.array(spin_states)
   
    return energies_MHz, transition_elements, spin_states

def resonances_no_fermi(B_lab):
    #B_lab = np.array([Bx,By,Bz])
    n=4
    B_nv,B_nv_x,B_nv_y,B_nv_z = nv_components(B_lab)
    vals,vecs = Hamiltonian(B_nv_x,B_nv_y,B_nv_z,n)
    index_to_remove = findzero(vecs)
    return res_values(vals,vecs,index_to_remove,n)





########################################################

# Define the path to your HDF5 file
h5_file_path = "Bulk_diamond_2600-3100MHz_sweep_2coils_sideshift_001/Bulk_diamond_2600-3100MHz_sweep_2coils_sideshift_00000.h5"
file = h5py.File(h5_file_path, "r")

# List all the groups and datasets in the HDF5 file
print("Groups and datasets in the HDF5 file:")
for name in file:
    print(name)
"""
def print_hdf5_item(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")  
# Recursively iterate through the HDF5 file to print groups and datasets
file.visititems(print_hdf5_item)

# Access a specific dataset
dataset_name = "000/dev18113/auxins/0/sample.auxin0.avg/chunkheader"  # Change to the name of your dataset
if dataset_name in file:
    dataset = file[dataset_name]
    # Read the data from the dataset into a NumPy array
    data = dataset[:]
    # You can now work with the 'data' array
    print("Data from the dataset:")
    print(data)
else:
    print(f"Dataset '{000}' not found in the HDF5 file.")
"""
numtest = "000"       
print(numtest+"/dev18113/demods/2/sample.x.avg/chunkheader")


def open_data(h5_file_path,*nums):
    file = h5py.File(h5_file_path, "r")
    data_dict={}
    nums = nums
    for i,num in enumerate(nums):
        name_of_set = file[num+"/dev18113/demods/2/sample.x.avg/chunkheader"][0][22]
        clean_name_of_set = str(name_of_set).replace("b","")
        clean_name_of_set = str(clean_name_of_set).replace("'","")
        dic={
        "777_X_t": file[str(num)+"/dev18113/demods/2/sample.x.avg/timestamp"][:],
        "777_X_v": file[str(num)+"/dev18113/demods/2/sample.x.avg/value"][:],
        "dt": file[str(num)+"/dev18113/auxins/0/sample.auxin0.avg"].attrs["timebase"],
        #"777_Y_t": file[str(num)+"/dev18113/demods/2/sample.y.avg/timestamp"][:],
        "777_Y_v": file[str(num)+"/dev18113/demods/2/sample.y.avg/value"][:],
        "976_X_t": file[str(num)+"/dev18113/demods/3/sample.x.avg/timestamp"][:],
        "976_X_v": file[str(num)+"/dev18113/demods/3/sample.x.avg/value"][:],
        #"_976_Y_t": file[str(num)+"/dev18113/demods/3/sample.y.avg/timestamp"][:],
        "976_Y_v": file[str(num)+"/dev18113/demods/3/sample.y.avg/value"][:],
        }
        print(clean_name_of_set)
        data_dict[str(clean_name_of_set)] = dic
    return data_dict
"""
def open_data_calibrate(h5_file_path,num):
    name_of_set = file[num+"/dev18113/demods/2/sample.x.avg/chunkheader"][0][22]
    print(name_of_set)
    D2_777_X_t = file[num+"/dev18113/demods/2/sample.x.avg/timestamp"][:]
    D2_777_X_v = file[num+"/dev18113/demods/2/sample.x.avg/value"][:]
    D2_777_Y_t =file[num+"/dev18113/demods/2/sample.y.avg/timestamp"][:]
    D2_777_Y_v = file[num+"/dev18113/demods/2/sample.y.avg/value"][:]
    D3_996_X_t = file[num+"/dev18113/demods/3/sample.x.avg/timestamp"][:]
    D3_996_X_v = file[num+"/dev18113/demods/3/sample.x.avg/value"][:]
    D3_996_Y_t = file[num+"/dev18113/demods/3/sample.y.avg/timestamp"][:]
    D3_996_Y_v = file[num+"/dev18113/demods/3/sample.y.avg/value"][:]
"""

# =============================================================================
# calibration = open_data("Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_000/Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_00000.h5","000")
# data_0cm = open_data("Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_000/Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_00000.h5","001","002","003")
# data_1cm = open_data("Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_000/Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_00000.h5","004","005","006")
# data_2cm = open_data("Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_000/Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_00000.h5","007","008","009")
# data_3cm = open_data("Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_000/Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_00000.h5","010","011","012")
# data_4cm = open_data("Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_000/Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_00000.h5","013","014","015")
# 
# fulldata = open_data("Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_000/Bulk_diamond_2600-3100MHz_sweep_2coils_backshift_00000.h5","000","001","002","003","004","005","006","007","008","009","010","011","012","013","014","015")
# 
# =============================================================================
"""
#####RAW PLOTS START
for num in [0,1,2,3,4]:
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],fulldata['FM_777and976coils_'+str(num)+'cm_1']['777_X_v'][:],label="1")
    ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],fulldata['FM_777and976coils_'+str(num)+'cm_2']['777_X_v'][:],label="2")
    ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],fulldata['FM_777and976coils_'+str(num)+'cm_3']['777_X_v'][:],label="3")
            #,
            #np.mean(data_1cm['FM_777and976coils_0cm_1']['777_Y_v'][:],data_1cm['FM_777and976coils_0cm_2']['777_Y_v'][:],data_1cm['FM_777and976coils_0cm_3']['777_Y_v'][:]))
    ax.set_title(str(num)+'cm translation 777')
    fig.legend(bbox_to_anchor=(1,1),fontsize=16)
    fig.tight_layout()
    fig.savefig(str(num)+'cm_translation_back_777.pdf')
    
    
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['976_X_t'][:],fulldata['FM_777and976coils_'+str(num)+'cm_1']['976_X_v'][:],label="1")
    ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['976_X_t'][:],fulldata['FM_777and976coils_'+str(num)+'cm_2']['976_X_v'][:],label="2")
    ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['976_X_t'][:],fulldata['FM_777and976coils_'+str(num)+'cm_3']['976_X_v'][:],label="3")
            #,
            #np.mean(data_1cm['FM_777and976coils_0cm_1']['777_Y_v'][:],data_1cm['FM_777and976coils_0cm_2']['777_Y_v'][:],data_1cm['FM_777and976coils_0cm_3']['777_Y_v'][:]))
    ax.set_title(str(num)+'cm translation 976')
    fig.legend(bbox_to_anchor=(1,1),fontsize=16)
    fig.tight_layout()
    fig.savefig(str(num)+'cm_translation_back_976.pdf')
    
"""
    
# =============================================================================
# #####MEAN PLOTS START
# means = []
# for num in [0,1,2,3,4]:
#     means.append(np.mean(np.array([fulldata['FM_777and976coils_'+str(num)+'cm_1']['777_X_v'][:],fulldata['FM_777and976coils_'+str(num)+'cm_2']['777_X_v'][:],fulldata['FM_777and976coils_'+str(num)+'cm_3']['777_X_v'][:]]),axis=0))
# # =============================================================================
# #         fig,ax = plt.subplots(1,1,figsize=(10,5))
# #         ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],means[num])
# #         ax.set_title(str(num)+'cm translation mean')
# #         fig.tight_layout()
# #         fig.savefig(str(num)+'cm_translation_back_mean.pdf')
# # =============================================================================
# 
# #####Plot of all means
# fig,ax=plt.subplots(1,1,figsize=(10,5))
# for i,mean in enumerate(means):
#     ax.plot(fulldata['FM_777and976coils_1cm_1']['777_X_t'][:],means[i],label=str(i)+"cm")
# ax.set_title("Mean FM signals due to translation 777")
# ax.legend(bbox_to_anchor=(1,1),fontsize=16)
# fig.tight_layout()
# fig.savefig('plot_of_means_back_777.pdf')
# 
# means = []
# for num in [0,1,2,3,4]:
#     means.append(np.mean(np.array([fulldata['FM_777and976coils_'+str(num)+'cm_1']['976_X_v'][:],fulldata['FM_777and976coils_'+str(num)+'cm_2']['976_X_v'][:],fulldata['FM_777and976coils_'+str(num)+'cm_3']['976_X_v'][:]]),axis=0))
# # =============================================================================
# #         fig,ax = plt.subplots(1,1,figsize=(10,5))
# #         ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],means[num])
# #         ax.set_title(str(num)+'cm translation mean')
# #         fig.tight_layout()
# #         fig.savefig(str(num)+'cm_translation_back_mean.pdf')
# # =============================================================================
# 
# #####Plot of all means
# fig,ax=plt.subplots(1,1,figsize=(10,5))
# for i,mean in enumerate(means):
#     ax.plot(fulldata['FM_777and976coils_1cm_1']['976_X_t'][:],means[i],label=str(i)+"cm")
# ax.set_title("Mean FM signals due to translation 976")
# ax.legend(bbox_to_anchor=(1,1),fontsize=16)
# fig.tight_layout()
# fig.savefig('plot_of_means_back_976.pdf')
# =============================================================================



fulldata_side = open_data("Bulk_diamond_2600-3100MHz_sweep_2coils_sideshift_000/Bulk_diamond_2600-3100MHz_sweep_2coils_sideshift_00000.h5","000","001","002","003","004","005","006","007","008","009","010","011","012","013","014","015")
#####RAW PLOTS START
"""
for num in [0,1,2,3,4]:
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(fulldata_side['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],fulldata_side['FM_777and976coils_'+str(num)+'cm_1']['777_X_v'][:],label="1")
    ax.plot(fulldata_side['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],fulldata_side['FM_777and976coils_'+str(num)+'cm_2']['777_X_v'][:],label="2")
    ax.plot(fulldata_side['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],fulldata_side['FM_777and976coils_'+str(num)+'cm_3']['777_X_v'][:],label="3")
    ax.set_title(str(num)+'cm translation')
    fig.legend(bbox_to_anchor=(1,1),fontsize=16)
    fig.tight_layout()
    fig.savefig(str(num)+'cm_translation_side.pdf')
    """
    
    
dt = fulldata_side['FM_777and976coils_'+str(0)+'cm_1']["dt"]
    
    
    
    
    
nb = 4
    
    
    
#######################################################
#################FOR SIDE SHIFT
########################################################


#ALL DEFINITIONS

#This defines the x axis for the measured spectra
freq_t = (fulldata_side['FM_777and976coils_1cm_1']['777_X_t']-fulldata_side['FM_777and976coils_1cm_1']['777_X_t'][0]) * dt/(80e-3) + 2600  




   


#Get the data!
def getmagneticfield_side_Yfit(lf,fwhm_gen, fwhm_fixed, factor_gen,factor_fixed,peak_threshold_gen, peak_threshold_fixed):
    means_side_x = []
    for num in range(nb):
        means_side_x.append(np.mean(np.array([fulldata_side['FM_777and976coils_'+str(num)+'cm_1'][str(lf)+'_X_v'][:],fulldata_side['FM_777and976coils_'+str(num)+'cm_2'][str(lf)+'_X_v'][:],fulldata_side['FM_777and976coils_'+str(num)+'cm_3'][str(lf)+'_X_v'][:]]),axis=0))
    means_side_x=np.array(means_side_x)
    # =============================================================================
    #         fig,ax = plt.subplots(1,1,figsize=(10,5))
    #         ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],means_side_x[num])
    #         ax.set_title(str(num)+'cm translation mean X')
    #         fig.tight_layout()
    #         fig.savefig(str(num)+'cm_translation_side_mean_x.pdf')
    # =============================================================================

    means_side_y = []
    for num in range(nb):
        means_side_y.append(np.mean(np.array([fulldata_side['FM_777and976coils_'+str(num)+'cm_1'][str(lf)+'_Y_v'][:],fulldata_side['FM_777and976coils_'+str(num)+'cm_2'][str(lf)+'_Y_v'][:],fulldata_side['FM_777and976coils_'+str(num)+'cm_3'][str(lf)+'_Y_v'][:]]),axis=0))
    means_side_y=np.array(means_side_y)
    # =============================================================================
    #         fig,ax = plt.subplots(1,1,figsize=(10,5))
    #         ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],means_side_y[num])
    #         ax.set_title(str(num)+'cm translation mean Y')
    #         #fig.tight_layout()
    #         #fig.savefig(str(num)+'cm_translation_side_mean_y.pdf')
    # =============================================================================

    colors =['k','b','green','indigo','deeppink','orange']

    # =============================================================================
    #     #####Plot of all means
    #     fig,ax=plt.subplots(1,1,figsize=(10,5))
    #     ax.plot(fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t'],fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'], c=colors[0],label='3MHz cal. X')
    #     ax.plot(fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t'],fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'], c=colors[0],label='3MHz cal. Y')
    #     for i,mean in enumerate(means_side_x):
    #         ax.plot(fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t'][:],means_side_x[i],label=str(i)+"cm X",c=colors[1:][i])
    #         ax.plot(fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t'][:],means_side_y[i],label=str(i)+"cm Y",c=colors[1:][i])
    #     ax.set_title("Mean FM signals due to translation")
    #     ax.legend(bbox_to_anchor=(1,1),fontsize=16)
    #     fig.tight_layout()
    #     fig.savefig(str(lf)+'plot_of_means_side.pdf')
    # =============================================================================

    freq_t = (fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t']-fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t'][0])*dt/(40e-3) + 2600 

    #Remove the medians: 
        #for X
    means_side_corr_x=[]
    for mean in means_side_x:
        mean_corr = mean - np.median(mean[:200])
        means_side_corr_x.append(mean_corr)
    means_side_corr_x=np.array(means_side_corr_x)
    #For Y:
    means_side_corr_y=[]
    for mean in means_side_y:
        mean_corr = mean - np.median(mean[:200])
        means_side_corr_y.append(mean_corr)
    means_side_corr_y=np.array(means_side_corr_y)

    #Plot the corrected means:
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'], c=colors[0],label='3MHz cal. X')
    ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'], c=colors[0],label='3MHz cal. Y')
    for i,mean in enumerate(means_side_corr_x):
        ax.plot(freq_t,means_side_corr_x[i],label=str(i)+"cm X",c=colors[1:][i])
        #ax.plot(freq_t,means_side_corr_y[i],label=str(i)+"cm Y",c=colors[1:][i])
    ax.set_title("Corrected mean FM signals due to translation "+str(lf))
    ax.set_xlabel('MW frequency (MHz)')
    ax.legend(bbox_to_anchor=(1,1),fontsize=16)
    fig.tight_layout()
    fig.savefig(str(lf)+'plot_of_corrected_means_side.pdf')





    def found_peaks(data_array, height):
        info = sp.signal.find_peaks(data_array, height = height)
        if info[0].shape[0] == 8:
            return info[0],info[1]['peak_heights']
        if info[0].shape[0] > 8:
            ###SORT THEM BY ASCENDING ORDER SO YOU CAN TAKE THE TOP 8.
            return info[0][np.argsort(info[1]['peak_heights'])][-8:], np.sort(info[1]['peak_heights'])[-8:]
        if info[0].shape[0] < 8:
            print("Thresholds were too high to detect all 8 peaks for " +str(lf))
            

    #Fit the 8 Gaussian derivatives to the 3MHz data. Obtain the true centres of the derivatives.
    def gaussian_deriv(theta,x):
        A, mu, sig = theta
        return -A*(1/(np.sqrt(2*np.pi)* sig**3))*(x-mu)*np.exp(-(x-mu)**2 / (2 * sig**2))

    def gaussian_8_deriv(x,A1, mu1, sig1,A2, mu2, sig2, A3, mu3, sig3,A4, mu4, sig4,A5, mu5, sig5,A6, mu6, sig6,A7, mu7, sig7,A8, mu8, sig8):
        theta1=[A1, mu1, sig1]
        theta2=[A2, mu2, sig2]
        theta3=[A3, mu3, sig3]
        theta4=[A4, mu4, sig4]
        theta5=[A5, mu5, sig5]
        theta6=[A6, mu6, sig6]
        theta7 =[A7, mu7, sig7]
        theta8 =[A8, mu8, sig8]
        return gaussian_deriv(theta1,x)+gaussian_deriv(theta2,x)+gaussian_deriv(theta3,x)+gaussian_deriv(theta4,x)+gaussian_deriv(theta5,x)+gaussian_deriv(theta6,x)+gaussian_deriv(theta7,x)+gaussian_deriv(theta8,x)


    def initial_guesses(data_array_X, data_array_Y,height):
        initial_guesses=[]
        for i in range(8):
            info = found_peaks(data_array_X,height)
            infoY = found_peaks(data_array_Y,height)
            initial_guesses.append(data_array_X[info[0][i]]/ np.cos(np.arctan(data_array_Y[infoY[0][i]]/data_array_X[info[0][i]])) * factor_gen)
            initial_guesses.append(freq_t[info[0][i]]+4)
            initial_guesses.append(fwhm_gen)
        initial_guesses.append(np.arctan(data_array_Y[found_peaks(data_array_Y,height)[0][0]]/data_array_X[info[0][0]]))
        return np.array(initial_guesses)



    #------- Optimisation ---------
    Xcal = fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v']
    Ycal= fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v']
    X = means_side_corr_x
    Y = means_side_corr_y

    def both_XY_gaussian_deriv(p,freq_t):    
        A1, mu1, sig1,A2, mu2, sig2, A3, mu3, sig3,A4, mu4, sig4,A5, mu5, sig5,A6, mu6, sig6,A7, mu7, sig7,A8, mu8, sig8 = p[:-1]
        p_phase = p[-1]
        signal = gaussian_8_deriv(freq_t,A1, mu1, sig1,A2, mu2, sig2, A3, mu3, sig3,A4, mu4, sig4,A5, mu5, sig5,A6, mu6, sig6,A7, mu7, sig7,A8, mu8, sig8)
        Xguess = signal*np.cos(p_phase)
        Yguess = signal*np.sin(p_phase)
        return Xguess,Yguess

    def lp_both_XY_gaussian_derivatives(p,freq_t, X,Y):
        A1, mu1, sig1,A2, mu2, sig2, A3, mu3, sig3,A4, mu4, sig4,A5, mu5, sig5,A6, mu6, sig6,A7, mu7, sig7,A8, mu8, sig8, p_phase=p
        Xguess,Yguess = both_XY_gaussian_deriv(p,freq_t)
        return np.sum((X-Xguess)**2) + np.sum((Y-Yguess)**2)


    #optimise
    initial_guess = initial_guesses(fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'],fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'],peak_threshold_gen)
    args = (freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'],fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'])

    result = minimize(lp_both_XY_gaussian_derivatives,
                         initial_guess,
                         args=args,
                         method="SLSQP",
                         )
    fitted_vars_all=result.x #Last one is phase
    fitted_gauss_params=fitted_vars_all[:-1]
    fitted_phase = fitted_vars_all[-1]
    fitted_X_cal = gaussian_8_deriv(freq_t,*fitted_gauss_params) * np.cos(fitted_phase)
    fitted_Y_cal = gaussian_8_deriv(freq_t,*fitted_gauss_params) * np.sin(fitted_phase)

    print('\n FITTED PHASE '+ str(lf) + ': ', fitted_phase, '\n')




    #THIS IS ALL FOR 3MHZ X


    #Plot the fits!
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'], c=colors[0],label='3MHz cal. X')
    #ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'], c=colors[0],label='3MHz cal. Y')
    ax.plot(freq_t,fitted_X_cal,c="r",label="X fit", linestyle="dashed")
    ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'], c=colors[1],label='3MHz cal. Y')
    #ax.plot(freq_t,fitted_X_cal,c="r",label="X fit")
    ax.plot(freq_t,fitted_Y_cal,c="green",label="Y fit", linestyle = "dashed")
    #ax.set_title("Corrected mean FM signals fitted Y")
    #ax.plot(freq_t,fitted_Y_cal,c="b",label="Y fit")
    #ax.set_title("Corrected mean FM signals fitted X")
    ax.set_xlabel('MW frequency (MHz)')
    ax.set_ylabel('FM signal (V)')
    ax.legend(bbox_to_anchor=(1,1),fontsize=16)
    fig.tight_layout()
    fig.savefig(str(lf)+'plot_of_corrected_calibration_side_x_FITTED.pdf')

    fig,ax=plt.subplots(1,1,figsize=(10,5))
    #ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'], c=colors[0],label='3MHz cal. X')
    ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'], c=colors[0],label='3MHz cal. Y')
    #ax.plot(freq_t,fitted_X_cal,c="r",label="X fit")
    ax.plot(freq_t,fitted_Y_cal,c="b",label="Y fit")
    #ax.set_title("Corrected mean FM signals fitted Y")
    ax.set_xlabel('MW frequency (MHz)')
    ax.set_ylabel('FM signal (V)')
    ax.legend(bbox_to_anchor=(1,1),fontsize=16)
    fig.tight_layout()
    fig.savefig(str(lf)+'plot_of_corrected_calibration_side_y_FITTED.pdf')

    peak_indices_unsorted = found_peaks(Xcal,peak_threshold_gen)[0]
    peak_indices = np.sort(peak_indices_unsorted)
    fixed_resonances = np.array([fitted_gauss_params[i] for i in [1,4,7,10,13,16,19,22]])
    unsorted_fixed_fwhm = np.array([fitted_gauss_params[i] for i in [2, 5, 8, 11, 14, 17, 20, 23]])
    unordered_cal_amps = np.array([fitted_gauss_params[i] for i in [0,3,6,9,12,15,18,21]])
    ordered_cal_amps = unordered_cal_amps[np.argsort(fixed_resonances.flatten())]
    ordered_fixed_fwhm = unsorted_fixed_fwhm[np.argsort(fixed_resonances.flatten())]
    ordered_fixed_resonances = np.sort(fixed_resonances.flatten())
    flattened_fixed_resonances = ordered_fixed_resonances
    
    Dmean = np.mean([(flattened_fixed_resonances[i]+flattened_fixed_resonances[-i-1])/2 for i in range(4)])
    print('\n D MEAN : ', Dmean, '\n')
    print('\n CALIBRATION LINEWIDTHS '+ str(lf)+': ', ordered_fixed_fwhm)
    print('\n CALIBRATION AMPLITUDES '+ str(lf)+': ', ordered_cal_amps, '\n')
    print('RESONANCES '+ str(lf)+': ', flattened_fixed_resonances)


    ##########I WILL BE USING THE 'SIDE' DATA, NOT BACK
    #Apply 8 Gaussian fits with fixed centres. Obtain the amplitudes from the fit.

    def gaussian_deriv_fixed(theta,x):
        A, mu, sig = theta
        return -A*(1/(np.sqrt(2*np.pi)* sig**3))*(x-mu)*np.exp(-(x-mu)**2 / (2 * sig**2))

    def gaussian_deriv_fixed_single(theta,x, flattened_fixed_resonance):
        A, sig = theta
        mu = flattened_fixed_resonance
        return -A*(1/(np.sqrt(2*np.pi)* sig**3))*(x-mu)*np.exp(-(x-mu)**2 / (2 * sig**2))

    def gaussian_8_deriv_fixed(x,A1, sig1, A2, sig2, A3, sig3, A4, sig4, A5, sig5, A6, sig6, A7, sig7, A8, sig8):
        #sig1, sig2, sig3, sig4, sig5, sig6, sig7, sig8 = fixed_fwhm
        mu1,mu2,mu3,mu4,mu5,mu6,mu7,mu8 = flattened_fixed_resonances
        theta1=[A1, mu1, sig1]
        theta2=[A2, mu2, sig2]
        theta3=[A3, mu3, sig3]
        theta4=[A4, mu4, sig4]
        theta5=[A5, mu5, sig5]
        theta6=[A6, mu6, sig6]
        theta7 =[A7, mu7, sig7]
        theta8 =[A8, mu8, sig8]
        return gaussian_deriv(theta1,x)+gaussian_deriv(theta2,x)+gaussian_deriv(theta3,x)+gaussian_deriv(theta4,x)+gaussian_deriv(theta5,x)+gaussian_deriv(theta6,x)+gaussian_deriv(theta7,x)+gaussian_deriv(theta8,x)

    def initial_guesses_fixed(data_array_X,data_array_Y,height):
        initial_guesses=[]
        peak_indices_X_unsorted, peak_heights_X = found_peaks(data_array_X, height)
        peak_indices_Y, peak_heights_Y = found_peaks(data_array_Y**2, height**2)
        peak_indices_X=np.sort(peak_indices_X_unsorted)
        average_init_phase = np.mean(np.arctan(data_array_Y[peak_indices] / data_array_X[peak_indices]))
            
        if ((data_array_Y[peak_indices[1]] < 0) and (data_array_X[peak_indices[1]] > 0)):
            for index, _ in enumerate(peak_indices):
                initial_guesses.append(data_array_Y[peak_indices[index]] / np.sin(average_init_phase) * factor_fixed)
                initial_guesses.append(fwhm_fixed)

            initial_guesses.append(average_init_phase)
            
        elif ((data_array_Y[peak_indices[1]] > 0) and (data_array_X[peak_indices[1]] < 0)):
            for index, _ in enumerate(peak_indices):
                initial_guesses.append(data_array_Y[peak_indices[index]] / np.sin(average_init_phase) * factor_fixed)
                initial_guesses.append(fwhm_fixed)
            initial_guesses.append(average_init_phase)
                
        else:
            for index, _ in enumerate(peak_indices):
                initial_guesses.append(data_array_Y[peak_indices[index]] / np.sin(average_init_phase) * factor_fixed)
                initial_guesses.append(fwhm_fixed)
        #for index, _ in enumerate(peak_indices):
            #initial_guesses.append(data_array_X[peak_indices[index]] * np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]]) * factor_fixed)
            #initial_guesses.append(np.array(fixed_fwhm[index]))
            initial_guesses.append(average_init_phase)
    # =============================================================================
    #         if ((data_array_Y[peak_indices[0]] < 0) and (data_array_X[peak_indices[0]] > 0)):
    #             for index, _ in enumerate(peak_indices):
    #                 initial_guesses.append(A_guesses[index] / np.cos(np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]])) * factor_fixed)
    #                 #initial_guesses.append(fwhm_fixed)
    #     
    #             initial_guesses.append(np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]]))
    #             
    #         elif ((data_array_Y[peak_indices[0]] > 0) and (data_array_X[peak_indices[0]] < 0)):
    #             for index, _ in enumerate(peak_indices):
    #                 initial_guesses.append(A_guesses[index] / np.cos(np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]])) * factor_fixed)
    #                 #initial_guesses.append(fwhm_fixed)
    #             initial_guesses.append(np.pi+np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]]))
    #                 
    #         else:
    #             for index, _ in enumerate(peak_indices):
    #                 initial_guesses.append(A_guesses[index] / np.cos(np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]])) * factor_fixed)
    #                 #initial_guesses.append(fwhm_fixed)
    #             initial_guesses.append(np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]]))
    # =============================================================================



    # =============================================================================
    #             for i,_ in enumerate(found_peaks(data_array_X,height)[0]):
    #                 info = found_peaks(data_array_X,height)
    #                 infoY = found_peaks(data_array_Y,height)
    #                 initial_guesses.append(data_array_X[info[0][i]]*np.arctan(data_array_Y[infoY[0][i]]/data_array_X[info[0][i]]))
    #                 initial_guesses.append(fwhm_fixed)
    #             initial_guesses.append(np.arctan(data_array_Y[found_peaks(data_array_Y,height)[0][0]]/data_array_X[info[0][0]]))      
    # =============================================================================
        return np.array(initial_guesses)

    def both_XY_gaussian_deriv_fixed(p,freq_t,flattened_fixed_resonances):    
        A1,sig1, A2, sig2, A3, sig3, A4, sig4, A5, sig5, A6, sig6, A7, sig7, A8, sig8= p[:-1]
        p_phase = p[-1]
        mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8 = flattened_fixed_resonances
        #sig1, sig2, sig3, sig4, sig5, sig6, sig7, sig8 = fixed_fwhm
        signal = gaussian_8_deriv(freq_t,A1, mu1, sig1,A2, mu2, sig2, A3, mu3, sig3,A4, mu4, sig4,A5, mu5, sig5,A6, mu6, sig6,A7, mu7, sig7,A8, mu8, sig8)
        Xguess = signal*np.cos(p_phase)
        Yguess = signal*np.sin(p_phase)
        return Xguess,Yguess

    def lp_both_XY_gaussian_derivatives_fixed(p, freq_t, X,Y, flattened_fixed_resonances):
        Xguess,Yguess = both_XY_gaussian_deriv_fixed(p,freq_t,flattened_fixed_resonances)
        return np.sum((X-Xguess)**2) + np.sum((Y-Yguess)**2)


    #Optimise
    def fitfunc_param_fixed(initial_guess_fixed, freq_t, X, Y):
        initial_guess = initial_guess_fixed
        args = (freq_t,X,Y,flattened_fixed_resonances)
        
        
        if ((Y[peak_indices[1]] > 0) and (X[peak_indices[1]] < 0)):
            result = minimize(lp_both_XY_gaussian_derivatives_fixed,
                             initial_guess,
                             args=args,
                             method="SLSQP",
                             bounds = [
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (-np.pi/2, 0),
                                 ]
                             )
        elif ((Y[peak_indices[1]] < 0) and (X[peak_indices[1]] > 0)):
            result = minimize(lp_both_XY_gaussian_derivatives_fixed,
                             initial_guess,
                             args=args,
                             method="SLSQP",
                             bounds = [
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (-np.pi/2, 0),
                                 ]
                             )
        else:
            result = minimize(lp_both_XY_gaussian_derivatives_fixed,
                             initial_guess,
                             args=args,
                             method="SLSQP",
                             bounds = [
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (0, np.pi/2),
                                 ]
                             )
        fitted_vars_all_fixed=result.x #Last one is phase
        
        fitted_curve = gaussian_8_deriv_fixed(freq_t,*fitted_vars_all_fixed[:-1])
        
        # Plot the original data and the fitted curves
        
        fig,ax=plt.subplots(1,1,figsize=(10,5))
        ax.plot(freq_t, X, c="k",label = 'X Data')
        ax.plot(freq_t, fitted_curve*np.cos(fitted_vars_all_fixed[-1]), 'r', label='X Fit')
        ax.plot(freq_t, Y, c="b",label = 'Y Data')
        ax.plot(freq_t, fitted_curve*np.sin(fitted_vars_all_fixed[-1]), 'purple', label='Y Fit')
        ax.set_ylabel('FM signal (V)')
        ax.set_xlabel('MW frequency (MHz)')
        ax.legend(bbox_to_anchor=(1,1),fontsize=16)
        fig.tight_layout()
        fig.savefig("777_example.pdf")
        
        
        return fitted_vars_all_fixed


    #FOR X DATA 777Hz
    all_params_fixed = []
    fitted_phase_fixed=[]
    fitted_gauss_params_fixed=[]
    fitted_X_fixed = []
    fitted_Y_fixed = []

    for ind,mean in enumerate(X):
        initial_guesses_x_fixed = initial_guesses_fixed(X[ind],Y[ind],peak_threshold_fixed)
        fitted_vars_fixed=fitfunc_param_fixed(initial_guesses_x_fixed,freq_t,X[ind],Y[ind])
        fitted_phase_fixed_i=fitted_vars_fixed[-1]
        fitted_gauss_params_fixed_i = fitted_vars_fixed[:-1]
        all_params_fixed.append(fitted_vars_fixed)
        fitted_phase_fixed.append(fitted_phase_fixed_i)
        fitted_gauss_params_fixed.append(fitted_gauss_params_fixed_i)
        fitted_X_fixed.append(gaussian_8_deriv_fixed(freq_t,*fitted_gauss_params_fixed_i)*np.cos(fitted_phase_fixed_i))
        fitted_Y_fixed.append(gaussian_8_deriv_fixed(freq_t,*fitted_gauss_params_fixed_i)*np.sin(fitted_phase_fixed_i))
          

    fitted_phase_fixed = np.array(fitted_phase_fixed)
    fitted_gauss_params_fixed=np.array(fitted_gauss_params_fixed)
    all_params_fixed=np.array(all_params_fixed) #last will be phase again
    fitted_X_fixed = np.array(fitted_X_fixed)
    fitted_Y_fixed = np.array(fitted_Y_fixed)
        
    #Plot the fits!
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    for i,mean in enumerate(X):
        ax.plot(freq_t,X[i],label=str(i)+"cm X",c=colors[1:][i])
        ax.plot(freq_t,Y[i],label=str(i)+"cm Y",c=colors[1:][i])
        ax.plot(freq_t,fitted_X_fixed[i],c="r",label="X fit "+str(i))
        ax.plot(freq_t,fitted_Y_fixed[i],c="k",label="Y fit "+str(i))
    ax.set_title("Corrected mean FM signals fitted")
    ax.set_xlabel('MW frequency (MHz)')
    ax.set_ylabel('FM signal (V)')
    ax.legend(bbox_to_anchor=(1,1),fontsize=16)
    fig.tight_layout()
    fig.savefig('plot_of_corrected_side_FITTED.pdf')


    amps = fitted_gauss_params_fixed[:,::2]


    rel_amps = np.array([np.divide(amps[i],ordered_cal_amps) for i in range(nb)])

    rel_amps_MHz=rel_amps*3

    amp_pairs=np.zeros((5,4,2))
    for i, amplist in enumerate(rel_amps_MHz):
        for ind in range(4):
            amp_pairs[i,ind]=np.array([rel_amps_MHz[i,ind],rel_amps_MHz[i,-(ind+1)]])
        

    #Allocate the resonances as pairs:
    final_fixed_resonances=[]
    for i in range(4):
        sorted_fixed_resonances = np.sort(fixed_resonances)
        final_fixed_resonances.append(np.array([sorted_fixed_resonances[i],sorted_fixed_resonances[-(i+1)]]))
    final_fixed_resonances=np.array(final_fixed_resonances)
        


    ############################################
    #Define finding B_lab given function.

    def lp_fixed(theta,measured_resonances):
        Bx,By, Bz = theta
        B_lab = np.array([Bx,By,Bz])
        measured_resonances=np.sort(measured_resonances.flatten())
        residuals = measured_resonances - np.sort(resonances_fermi(B_lab, D=Dmean)[0].flatten())
        residual_variance = np.sum(residuals**2)
        return residual_variance


    #Testing on the fixed resonances to find the bias field.
    initial_guess = [-0.15,-0.1,-0.1]
    args = (final_fixed_resonances)

    result = minimize(lp_fixed,
                         initial_guess,
                         args=args,
                         method="SLSQP",
                         )
    fitted_B_lab=result.x



    ####The result is: fitted_B_lab
    #               Out[783]: array([-0.08357434, -0.00016348, -0.00011569])
    #This makes sense because we used a bar magnet, so the lines at the diamond would be pretty much going in one direction! Makes sense to have a much bigger component in one of the directions.


    #Test with first, 0cm, using known centres of each resonance
    def spin_resonance_AC_modulation(fitted_B_lab, Bac):
        return resonances_fermi(fitted_B_lab + Bac,D=Dmean)[0] - resonances_fermi(fitted_B_lab - Bac,D=Dmean)[0]

    def lp_AC(theta, measured_resonances,fitted_B_lab):
        Bac = theta
        amp_resonances= np.sort(measured_resonances.flatten())
        residuals=amp_resonances - np.sort(spin_resonance_AC_modulation(fitted_B_lab, Bac).flatten())
        residual_variance = np.sum(residuals**2)
        return residual_variance
        
    initial_guess = [-0.005,-0.004,0.0003]
    args = (amp_pairs[0],fitted_B_lab)

    result = minimize(lp_AC,
                         initial_guess,
                         args=args,
                         method="SLSQP",
                         )
    AC_fitted_B_lab=result.x

    #We need to know the strength of the coils AT the coils.


    def AC_field_mag_all_translations(list_of_5_sweeps):
        means_side_corr_x_local = list_of_5_sweeps
        results=[]
        for i,means in enumerate(means_side_corr_x_local):
            initial_guess = [-0.005,0.04,0.03]
            args = (amp_pairs[i],fitted_B_lab)

            result = minimize(lp_AC,
                                 initial_guess,
                                 args=args,
                                 method="SLSQP",
                                 )
            AC_fitted_B_lab=result.x
            print("actual measured freqs = ", amp_pairs[i])
            print("\n fitted AC Bs = ", AC_fitted_B_lab)
            results.append(AC_fitted_B_lab)
        results=np.array(results)
        return results

    mag_field_all = AC_field_mag_all_translations(means_side_corr_x)
    print("\n ALL MAGNETIC FIELDS " + str(lf) + ": ",   mag_field_all, '\n')

    abs_mag_field_all = []
    for i, mag_field in enumerate(mag_field_all):
        abs_mag_field_all.append(np.array(np.sqrt(mag_field[0]**2+mag_field[1]**2 + mag_field[2]**2)))
    abs_mag_field_all = np.array(abs_mag_field_all)
    #Find the differences between them and see if it corresponds to a particular direction!

    #diff_fields = np.array([(mag_field_all[i] - mag_field_all[0]) for i,v in enumerate(mag_field_all)])

    abs_diff_B_lab = [abs_mag_field_all[i]-abs_mag_field_all[0] for i,_ in enumerate(abs_mag_field_all)]
    
    return mag_field_all, abs_mag_field_all, abs_diff_B_lab









def getmagneticfield_side_Xfit(lf,fwhm_gen, fwhm_fixed, factor_gen,factor_fixed,peak_threshold_gen, peak_threshold_fixed):
    means_side_x = []
    for num in range(nb):
        means_side_x.append(np.mean(np.array([fulldata_side['FM_777and976coils_'+str(num)+'cm_1'][str(lf)+'_X_v'][:],fulldata_side['FM_777and976coils_'+str(num)+'cm_2'][str(lf)+'_X_v'][:],fulldata_side['FM_777and976coils_'+str(num)+'cm_3'][str(lf)+'_X_v'][:]]),axis=0))
    means_side_x=np.array(means_side_x)
    # =============================================================================
    #         fig,ax = plt.subplots(1,1,figsize=(10,5))
    #         ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],means_side_x[num])
    #         ax.set_title(str(num)+'cm translation mean X')
    #         fig.tight_layout()
    #         fig.savefig(str(num)+'cm_translation_side_mean_x.pdf')
    # =============================================================================

    means_side_y = []
    for num in range(nb):
        means_side_y.append(np.mean(np.array([fulldata_side['FM_777and976coils_'+str(num)+'cm_1'][str(lf)+'_Y_v'][:],fulldata_side['FM_777and976coils_'+str(num)+'cm_2'][str(lf)+'_Y_v'][:],fulldata_side['FM_777and976coils_'+str(num)+'cm_3'][str(lf)+'_Y_v'][:]]),axis=0))
    means_side_y=np.array(means_side_y)
    # =============================================================================
    #         fig,ax = plt.subplots(1,1,figsize=(10,5))
    #         ax.plot(fulldata['FM_777and976coils_'+str(num)+'cm_1']['777_X_t'][:],means_side_y[num])
    #         ax.set_title(str(num)+'cm translation mean Y')
    #         #fig.tight_layout()
    #         #fig.savefig(str(num)+'cm_translation_side_mean_y.pdf')
    # =============================================================================

    colors =['k','b','green','indigo','deeppink','orange']

    # =============================================================================
    #     #####Plot of all means
    #     fig,ax=plt.subplots(1,1,figsize=(10,5))
    #     ax.plot(fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t'],fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'], c=colors[0],label='3MHz cal. X')
    #     ax.plot(fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t'],fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'], c=colors[0],label='3MHz cal. Y')
    #     for i,mean in enumerate(means_side_x):
    #         ax.plot(fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t'][:],means_side_x[i],label=str(i)+"cm X",c=colors[1:][i])
    #         ax.plot(fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t'][:],means_side_y[i],label=str(i)+"cm Y",c=colors[1:][i])
    #     ax.set_title("Mean FM signals due to translation")
    #     ax.legend(bbox_to_anchor=(1,1),fontsize=16)
    #     fig.tight_layout()
    #     fig.savefig(str(lf)+'plot_of_means_side.pdf')
    # =============================================================================

    freq_t = (fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t']-fulldata_side['FM_777and976coils_1cm_1'][str(lf)+'_X_t'][0])*dt/(40e-3) + 2600 

    #Remove the medians: 
        #for X
    means_side_corr_x=[]
    for mean in means_side_x:
        mean_corr = mean - np.median(mean[:200])
        means_side_corr_x.append(mean_corr)
    means_side_corr_x=np.array(means_side_corr_x)
    #For Y:
    means_side_corr_y=[]
    for mean in means_side_y:
        mean_corr = mean - np.median(mean[:200])
        means_side_corr_y.append(mean_corr)
    means_side_corr_y=np.array(means_side_corr_y)

    #Plot the corrected means:
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'], c=colors[0],label='3MHz cal. X')
    ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'], c=colors[0],label='3MHz cal. Y')
    for i,mean in enumerate(means_side_corr_x):
        ax.plot(freq_t,means_side_corr_x[i],label=str(i)+"cm X",c=colors[1:][i])
        #ax.plot(freq_t,means_side_corr_y[i],label=str(i)+"cm Y",c=colors[1:][i])
    ax.set_title("Corrected mean FM signals due to translation "+str(lf))
    ax.set_xlabel('MW frequency (MHz)')
    ax.legend(bbox_to_anchor=(1,1),fontsize=16)
    fig.tight_layout()
    fig.savefig(str(lf)+'plot_of_corrected_means_side.pdf')





    def found_peaks(data_array, height):
        info = sp.signal.find_peaks(data_array, height = height)
        if info[0].shape[0] == 8:
            return info[0],info[1]['peak_heights']
        if info[0].shape[0] > 8:
            ###SORT THEM BY ASCENDING ORDER SO YOU CAN TAKE THE TOP 8.
            return info[0][np.argsort(info[1]['peak_heights'])][-8:], np.sort(info[1]['peak_heights'])[-8:]
        if info[0].shape[0] < 8:
            print("Thresholds were too high to detect all 8 peaks for " +str(lf))
            

    #Fit the 8 Gaussian derivatives to the 3MHz data. Obtain the true centres of the derivatives.
    def gaussian_deriv(theta,x):
        A, mu, sig = theta
        return -A*(1/(np.sqrt(2*np.pi)* sig**3))*(x-mu)*np.exp(-(x-mu)**2 / (2 * sig**2))

    def gaussian_8_deriv(x,A1, mu1, sig1,A2, mu2, sig2, A3, mu3, sig3,A4, mu4, sig4,A5, mu5, sig5,A6, mu6, sig6,A7, mu7, sig7,A8, mu8, sig8):
        theta1=[A1, mu1, sig1]
        theta2=[A2, mu2, sig2]
        theta3=[A3, mu3, sig3]
        theta4=[A4, mu4, sig4]
        theta5=[A5, mu5, sig5]
        theta6=[A6, mu6, sig6]
        theta7 =[A7, mu7, sig7]
        theta8 =[A8, mu8, sig8]
        return gaussian_deriv(theta1,x)+gaussian_deriv(theta2,x)+gaussian_deriv(theta3,x)+gaussian_deriv(theta4,x)+gaussian_deriv(theta5,x)+gaussian_deriv(theta6,x)+gaussian_deriv(theta7,x)+gaussian_deriv(theta8,x)


    def initial_guesses(data_array_X, data_array_Y,height):
        initial_guesses=[]
        for i in range(8):
            info = found_peaks(data_array_X,height)
            infoY = found_peaks(data_array_Y,height)
            initial_guesses.append(data_array_X[info[0][i]]/ np.cos(np.arctan(data_array_Y[infoY[0][i]]/data_array_X[info[0][i]])) * factor_gen)
            initial_guesses.append(freq_t[info[0][i]]+4)
            initial_guesses.append(fwhm_gen)
        initial_guesses.append(np.arctan(data_array_Y[found_peaks(data_array_Y,height)[0][0]]/data_array_X[info[0][0]]))
        return np.array(initial_guesses)



    #------- Optimisation ---------
    Xcal = fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v']
    Ycal= fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v']
    X = means_side_corr_x
    Y = means_side_corr_y

    def both_XY_gaussian_deriv(p,freq_t):    
        A1, mu1, sig1,A2, mu2, sig2, A3, mu3, sig3,A4, mu4, sig4,A5, mu5, sig5,A6, mu6, sig6,A7, mu7, sig7,A8, mu8, sig8 = p[:-1]
        p_phase = p[-1]
        signal = gaussian_8_deriv(freq_t,A1, mu1, sig1,A2, mu2, sig2, A3, mu3, sig3,A4, mu4, sig4,A5, mu5, sig5,A6, mu6, sig6,A7, mu7, sig7,A8, mu8, sig8)
        Xguess = signal*np.cos(p_phase)
        Yguess = signal*np.sin(p_phase)
        return Xguess,Yguess

    def lp_both_XY_gaussian_derivatives(p,freq_t, X,Y):
        A1, mu1, sig1,A2, mu2, sig2, A3, mu3, sig3,A4, mu4, sig4,A5, mu5, sig5,A6, mu6, sig6,A7, mu7, sig7,A8, mu8, sig8, p_phase=p
        Xguess,Yguess = both_XY_gaussian_deriv(p,freq_t)
        return np.sum((X-Xguess)**2) + np.sum((Y-Yguess)**2)


    #optimise
    initial_guess = initial_guesses(fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'],fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'],peak_threshold_gen)
    args = (freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'],fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'])

    result = minimize(lp_both_XY_gaussian_derivatives,
                         initial_guess,
                         args=args,
                         method="SLSQP",
                         )
    fitted_vars_all=result.x #Last one is phase
    fitted_gauss_params=fitted_vars_all[:-1]
    fitted_phase = fitted_vars_all[-1]
    fitted_X_cal = gaussian_8_deriv(freq_t,*fitted_gauss_params) * np.cos(fitted_phase)
    fitted_Y_cal = gaussian_8_deriv(freq_t,*fitted_gauss_params) * np.sin(fitted_phase)

    print('\n FITTED PHASE '+ str(lf) + ': ', fitted_phase, '\n')




    #THIS IS ALL FOR 3MHZ X


    #Plot the fits!
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'], c=colors[0],label='3MHz cal. X')
    #ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'], c=colors[0],label='3MHz cal. Y')
    ax.plot(freq_t,fitted_X_cal,c="r",label="X fit", linestyle="dashed")
    #ax.plot(freq_t,fitted_Y_cal,c="b",label="Y fit")
    ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'], c=colors[1],label='3MHz cal. Y')
    #ax.plot(freq_t,fitted_X_cal,c="r",label="X fit")
    ax.plot(freq_t,fitted_Y_cal,c="green",label="Y fit",linestyle = "dashed")
    #ax.set_title("Corrected mean FM signals fitted X")
    ax.set_xlabel('MW frequency (MHz)')
    ax.set_ylabel('FM signal (V)')
    ax.legend(bbox_to_anchor=(1,1),fontsize=16)
    fig.tight_layout()
    fig.savefig(str(lf)+'plot_of_corrected_calibration_side_x_FITTED.pdf')

    fig,ax=plt.subplots(1,1,figsize=(10,5))
    #ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_X_v'], c=colors[0],label='3MHz cal. X')
    ax.plot(freq_t,fulldata_side['FM_3MHz_depth'][str(lf)+'_Y_v'], c=colors[0],label='3MHz cal. Y')
    #ax.plot(freq_t,fitted_X_cal,c="r",label="X fit")
    ax.plot(freq_t,fitted_Y_cal,c="b",label="Y fit")
    #ax.set_title("Corrected mean FM signals fitted Y")
    ax.set_xlabel('MW frequency (MHz)')
    ax.set_ylabel('FM signal (V)')
    ax.legend(bbox_to_anchor=(1,1),fontsize=16)
    fig.tight_layout()
    fig.savefig(str(lf)+'plot_of_corrected_calibration_side_y_FITTED.pdf')

    peak_indices_unsorted = found_peaks(Xcal,peak_threshold_gen)[0]
    peak_indices = np.sort(peak_indices_unsorted)
    fixed_resonances = np.array([fitted_gauss_params[i] for i in [1,4,7,10,13,16,19,22]])
    unsorted_fixed_fwhm = np.array([fitted_gauss_params[i] for i in [2, 5, 8, 11, 14, 17, 20, 23]])
    unordered_cal_amps = np.array([fitted_gauss_params[i] for i in [0,3,6,9,12,15,18,21]])
    ordered_cal_amps = unordered_cal_amps[np.argsort(fixed_resonances.flatten())]
    ordered_fixed_fwhm = unsorted_fixed_fwhm[np.argsort(fixed_resonances.flatten())]
    ordered_fixed_resonances = np.sort(fixed_resonances.flatten())
    flattened_fixed_resonances = ordered_fixed_resonances
    
    Dmean = np.mean([(flattened_fixed_resonances[i]+flattened_fixed_resonances[-i-1])/2 for i in range(4)])
    print('\n D MEAN : ', Dmean)
    print('\n CALIBRATION LINEWIDTHS '+ str(lf)+': ', ordered_fixed_fwhm)
    print('\n CALIBRATION AMPLITUDES '+ str(lf)+': ', ordered_cal_amps, '\n')
    print('RESONANCES '+ str(lf)+': ', flattened_fixed_resonances)


    ##########I WILL BE USING THE 'SIDE' DATA, NOT BACK
    #Apply 8 Gaussian fits with fixed centres. Obtain the amplitudes from the fit.

    def gaussian_deriv_fixed(theta,x):
        A, mu, sig = theta
        return -A*(1/(np.sqrt(2*np.pi)* sig**3))*(x-mu)*np.exp(-(x-mu)**2 / (2 * sig**2))

    def gaussian_deriv_fixed_single(theta,x, flattened_fixed_resonance):
        A, sig = theta
        mu = flattened_fixed_resonance
        return -A*(1/(np.sqrt(2*np.pi)* sig**3))*(x-mu)*np.exp(-(x-mu)**2 / (2 * sig**2))

    def gaussian_8_deriv_fixed(x,A1, sig1, A2, sig2, A3, sig3, A4, sig4, A5, sig5, A6, sig6, A7, sig7, A8, sig8):
        #sig1, sig2, sig3, sig4, sig5, sig6, sig7, sig8 = fixed_fwhm
        mu1,mu2,mu3,mu4,mu5,mu6,mu7,mu8 = flattened_fixed_resonances
        theta1=[A1, mu1, sig1]
        theta2=[A2, mu2, sig2]
        theta3=[A3, mu3, sig3]
        theta4=[A4, mu4, sig4]
        theta5=[A5, mu5, sig5]
        theta6=[A6, mu6, sig6]
        theta7 =[A7, mu7, sig7]
        theta8 =[A8, mu8, sig8]
        return gaussian_deriv(theta1,x)+gaussian_deriv(theta2,x)+gaussian_deriv(theta3,x)+gaussian_deriv(theta4,x)+gaussian_deriv(theta5,x)+gaussian_deriv(theta6,x)+gaussian_deriv(theta7,x)+gaussian_deriv(theta8,x)

    def initial_guesses_fixed(data_array_X,data_array_Y,height):
        initial_guesses=[]
        peak_indices_X_unsorted, peak_heights_X = found_peaks(data_array_X, height)
        peak_indices_Y, peak_heights_Y = found_peaks(data_array_Y**2, height**2)
        peak_indices_X=np.sort(peak_indices_X_unsorted)
        average_init_phase = np.mean(np.arctan(data_array_Y[peak_indices] / data_array_X[peak_indices]))
            
        if ((data_array_Y[peak_indices[1]] < 0) and (data_array_X[peak_indices[1]] > 0)):
            for index, _ in enumerate(peak_indices):
                initial_guesses.append(data_array_X[peak_indices[index]] / np.cos(average_init_phase) * factor_fixed)
                initial_guesses.append(fwhm_fixed)

            initial_guesses.append(average_init_phase)
            
        elif ((data_array_Y[peak_indices[1]] > 0) and (data_array_X[peak_indices[1]] < 0)):
            for index, _ in enumerate(peak_indices):
                initial_guesses.append(data_array_X[peak_indices[index]] / np.cos(average_init_phase) * factor_fixed)
                initial_guesses.append(fwhm_fixed)
            initial_guesses.append(average_init_phase)
                
        else:
            for index, _ in enumerate(peak_indices):
                initial_guesses.append(data_array_X[peak_indices[index]] / np.cos(average_init_phase) * factor_fixed)
                initial_guesses.append(fwhm_fixed)
        #for index, _ in enumerate(peak_indices):
            #initial_guesses.append(data_array_X[peak_indices[index]] * np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]]) * factor_fixed)
            #initial_guesses.append(np.array(fixed_fwhm[index]))
            initial_guesses.append(average_init_phase)
    # =============================================================================
    #         if ((data_array_Y[peak_indices[0]] < 0) and (data_array_X[peak_indices[0]] > 0)):
    #             for index, _ in enumerate(peak_indices):
    #                 initial_guesses.append(A_guesses[index] / np.cos(np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]])) * factor_fixed)
    #                 #initial_guesses.append(fwhm_fixed)
    #     
    #             initial_guesses.append(np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]]))
    #             
    #         elif ((data_array_Y[peak_indices[0]] > 0) and (data_array_X[peak_indices[0]] < 0)):
    #             for index, _ in enumerate(peak_indices):
    #                 initial_guesses.append(A_guesses[index] / np.cos(np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]])) * factor_fixed)
    #                 #initial_guesses.append(fwhm_fixed)
    #             initial_guesses.append(np.pi+np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]]))
    #                 
    #         else:
    #             for index, _ in enumerate(peak_indices):
    #                 initial_guesses.append(A_guesses[index] / np.cos(np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]])) * factor_fixed)
    #                 #initial_guesses.append(fwhm_fixed)
    #             initial_guesses.append(np.arctan(data_array_Y[peak_indices[0]] / data_array_X[peak_indices[0]]))
    # =============================================================================



    # =============================================================================
    #             for i,_ in enumerate(found_peaks(data_array_X,height)[0]):
    #                 info = found_peaks(data_array_X,height)
    #                 infoY = found_peaks(data_array_Y,height)
    #                 initial_guesses.append(data_array_X[info[0][i]]*np.arctan(data_array_Y[infoY[0][i]]/data_array_X[info[0][i]]))
    #                 initial_guesses.append(fwhm_fixed)
    #             initial_guesses.append(np.arctan(data_array_Y[found_peaks(data_array_Y,height)[0][0]]/data_array_X[info[0][0]]))      
    # =============================================================================
        return np.array(initial_guesses)

    def both_XY_gaussian_deriv_fixed(p,freq_t,flattened_fixed_resonances):    
        A1,sig1, A2, sig2, A3, sig3, A4, sig4, A5, sig5, A6, sig6, A7, sig7, A8, sig8= p[:-1]
        p_phase = p[-1]
        mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8 = flattened_fixed_resonances
        #sig1, sig2, sig3, sig4, sig5, sig6, sig7, sig8 = fixed_fwhm
        signal = gaussian_8_deriv(freq_t,A1, mu1, sig1,A2, mu2, sig2, A3, mu3, sig3,A4, mu4, sig4,A5, mu5, sig5,A6, mu6, sig6,A7, mu7, sig7,A8, mu8, sig8)
        Xguess = signal*np.cos(p_phase)
        Yguess = signal*np.sin(p_phase)
        return Xguess,Yguess

    def lp_both_XY_gaussian_derivatives_fixed(p, freq_t, X,Y, flattened_fixed_resonances):
        Xguess,Yguess = both_XY_gaussian_deriv_fixed(p,freq_t,flattened_fixed_resonances)
        return np.sum((X-Xguess)**2) + np.sum((Y-Yguess)**2)


    #Optimise
    def fitfunc_param_fixed(initial_guess_fixed, freq_t, X, Y):
        initial_guess = initial_guess_fixed
        args = (freq_t,X,Y,flattened_fixed_resonances)
        
        
        if ((Y[peak_indices[1]] > 0) and (X[peak_indices[1]] < 0)):
            result = minimize(lp_both_XY_gaussian_derivatives_fixed,
                             initial_guess,
                             args=args,
                             method="SLSQP",
                             bounds = [
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (-np.pi/2, 0),
                                 ]
                             )
        elif ((Y[peak_indices[1]] < 0) and (X[peak_indices[1]] > 0)):
            result = minimize(lp_both_XY_gaussian_derivatives_fixed,
                             initial_guess,
                             args=args,
                             method="SLSQP",
                             bounds = [
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (-np.pi/2, 0),
                                 ]
                             )
        else:
            result = minimize(lp_both_XY_gaussian_derivatives_fixed,
                             initial_guess,
                             args=args,
                             method="SLSQP",
                             bounds = [
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (0, np.pi/2),
                                 ]
                             )
        fitted_vars_all_fixed=result.x #Last one is phase
        
        fitted_curve = gaussian_8_deriv_fixed(freq_t,*fitted_vars_all_fixed[:-1])
        
        # Plot the original data and the fitted curves
        
        fig,ax=plt.subplots(1,1,figsize=(10,5))
        ax.plot(freq_t, X, c="k",label = 'X Data')
        ax.plot(freq_t, fitted_curve*np.cos(fitted_vars_all_fixed[-1]), 'r', label='X Fit')
        ax.plot(freq_t, Y, c="b",label = 'Y Data')
        ax.plot(freq_t, fitted_curve*np.sin(fitted_vars_all_fixed[-1]), 'purple', label='Y Fit')
        ax.legend(bbox_to_anchor=(1,1),fontsize=16)
        fig.tight_layout()
        fig.savefig("976_example.pdf")
        
        
        return fitted_vars_all_fixed


    #FOR X DATA 777Hz
    all_params_fixed = []
    fitted_phase_fixed=[]
    fitted_gauss_params_fixed=[]
    fitted_X_fixed = []
    fitted_Y_fixed = []

    for ind,mean in enumerate(X):
        initial_guesses_x_fixed = initial_guesses_fixed(X[ind],Y[ind],peak_threshold_fixed)
        fitted_vars_fixed=fitfunc_param_fixed(initial_guesses_x_fixed,freq_t,X[ind],Y[ind])
        fitted_phase_fixed_i=fitted_vars_fixed[-1]
        fitted_gauss_params_fixed_i = fitted_vars_fixed[:-1]
        all_params_fixed.append(fitted_vars_fixed)
        fitted_phase_fixed.append(fitted_phase_fixed_i)
        fitted_gauss_params_fixed.append(fitted_gauss_params_fixed_i)
        fitted_X_fixed.append(gaussian_8_deriv_fixed(freq_t,*fitted_gauss_params_fixed_i)*np.cos(fitted_phase_fixed_i))
        fitted_Y_fixed.append(gaussian_8_deriv_fixed(freq_t,*fitted_gauss_params_fixed_i)*np.sin(fitted_phase_fixed_i))
          

    fitted_phase_fixed = np.array(fitted_phase_fixed)
    fitted_gauss_params_fixed=np.array(fitted_gauss_params_fixed)
    all_params_fixed=np.array(all_params_fixed) #last will be phase again
    fitted_X_fixed = np.array(fitted_X_fixed)
    fitted_Y_fixed = np.array(fitted_Y_fixed)
        
    #Plot the fits!
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    for i,mean in enumerate(X):
        ax.plot(freq_t,X[i],label=str(i)+"cm X",c=colors[1:][i])
        ax.plot(freq_t,Y[i],label=str(i)+"cm Y",c=colors[1:][i])
        ax.plot(freq_t,fitted_X_fixed[i],c="r",label="X fit "+str(i))
        ax.plot(freq_t,fitted_Y_fixed[i],c="k",label="Y fit "+str(i))
    ax.set_title("Corrected mean FM signals fitted")
    ax.set_xlabel('MW frequency (MHz)')
    ax.legend(bbox_to_anchor=(1,1),fontsize=16)
    fig.tight_layout()
    fig.savefig('plot_of_corrected_side_FITTED.pdf')


    amps = fitted_gauss_params_fixed[:,::2]


    rel_amps = np.array([np.divide(amps[i],ordered_cal_amps) for i in range(nb)])

    rel_amps_MHz=rel_amps*3

    amp_pairs=np.zeros((5,4,2))
    for i, amplist in enumerate(rel_amps_MHz):
        for ind in range(4):
            amp_pairs[i,ind]=np.array([rel_amps_MHz[i,ind],rel_amps_MHz[i,-(ind+1)]])
        

    #Allocate the resonances as pairs:
    final_fixed_resonances=[]
    for i in range(4):
        sorted_fixed_resonances = np.sort(fixed_resonances)
        final_fixed_resonances.append(np.array([sorted_fixed_resonances[i],sorted_fixed_resonances[-(i+1)]]))
    final_fixed_resonances=np.array(final_fixed_resonances)
        


    ############################################
    #Define finding B_lab given function.

    def lp_fixed(theta,measured_resonances):
        Bx,By, Bz = theta
        B_lab = np.array([Bx,By,Bz])
        measured_resonances=np.sort(measured_resonances.flatten())
        residuals = measured_resonances - np.sort(resonances_fermi(B_lab,D=Dmean)[0].flatten())
        residual_variance = np.sum(residuals**2)
        return residual_variance


    #Testing on the fixed resonances to find the bias field.
    initial_guess = [-0.15,-0.1,-0.1]
    args = (final_fixed_resonances)

    result = minimize(lp_fixed,
                         initial_guess,
                         args=args,
                         method="SLSQP",
                         )
    fitted_B_lab=result.x

    print('BIAS FIELD : ', fitted_B_lab)

    ####The result is: fitted_B_lab
    #               Out[783]: array([-0.08357434, -0.00016348, -0.00011569])
    #This makes sense because we used a bar magnet, so the lines at the diamond would be pretty much going in one direction! Makes sense to have a much bigger component in one of the directions.


    #Test with first, 0cm, using known centres of each resonance
    def spin_resonance_AC_modulation(fitted_B_lab, Bac):
        return resonances_fermi(fitted_B_lab + Bac,D=Dmean)[0] - resonances_fermi(fitted_B_lab - Bac,D=Dmean)[0]

    def lp_AC(theta, measured_resonances,fitted_B_lab):
        Bac = theta
        amp_resonances= np.sort(measured_resonances.flatten())
        residuals=amp_resonances - np.sort(spin_resonance_AC_modulation(fitted_B_lab, Bac).flatten())
        residual_variance = np.sum(residuals**2)
        return residual_variance
        
    initial_guess = [-0.005,-0.004,0.0003]
    args = (amp_pairs[0],fitted_B_lab)

    result = minimize(lp_AC,
                         initial_guess,
                         args=args,
                         method="SLSQP",
                         )
    AC_fitted_B_lab=result.x

    #We need to know the strength of the coils AT the coils.


    def AC_field_mag_all_translations(list_of_5_sweeps):
        means_side_corr_x_local = list_of_5_sweeps
        results=[]
        for i,means in enumerate(means_side_corr_x_local):
            initial_guess = [-0.005,0.04,0.03]
            args = (amp_pairs[i],fitted_B_lab)

            result = minimize(lp_AC,
                                 initial_guess,
                                 args=args,
                                 method="SLSQP",
                                 )
            AC_fitted_B_lab=result.x
            print("actual measured freqs = ", amp_pairs[i])
            print("\n fitted AC Bs = ", AC_fitted_B_lab)
            results.append(AC_fitted_B_lab)
        results=np.array(results)
        return results

    mag_field_all = AC_field_mag_all_translations(means_side_corr_x)
    print("\n ALL MAGNETIC FIELDS " + str(lf) + ": ",   mag_field_all, '\n')

    abs_mag_field_all = []
    for i, mag_field in enumerate(mag_field_all):
        abs_mag_field_all.append(np.array(np.sqrt(mag_field[0]**2+mag_field[1]**2 + mag_field[2]**2)))
    abs_mag_field_all = np.array(abs_mag_field_all)
    #Find the differences between them and see if it corresponds to a particular direction!

    #diff_fields = np.array([(mag_field_all[i] - mag_field_all[0]) for i,v in enumerate(mag_field_all)])

    abs_diff_B_lab = [abs_mag_field_all[i]-abs_mag_field_all[0] for i,_ in enumerate(abs_mag_field_all)]
    
    return mag_field_all, abs_mag_field_all, abs_diff_B_lab




abs_Blab_0to4cm_777 = getmagneticfield_side_Yfit(777,5, 5,100,100,1e-7,0.1e-7)


print(abs_Blab_0to4cm_777)



#abs_Blab_0to4cm_777 = getmagneticfield_side(777,3,35,35,1e-7,0.1e-7, [1e-6, -0.5e-6, -0.5e-6, -0.5e-6, -0.5e-6, -0.7e-6, -0.3e-6, 0.5e-6])
abs_Blab_0to4cm_976 = getmagneticfield_side_Xfit(976,5, 5,100,100,1e-7,0.1e-7)
    

print(abs_Blab_0to4cm_976)
    
    
B_mag_0to4cm_777 = abs_Blab_0to4cm_777[1]
B_mag_0to4cm_976 = abs_Blab_0to4cm_976[1]
    
    
    

    
    
    
    
    #NOW FIND THE RADIUS FROM THE 777 COIL AND 976 COIL
    
    
   