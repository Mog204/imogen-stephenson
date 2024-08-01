# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:28:38 2023

@author: imy1
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 19:40:26 2023

@author: imy1
"""
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import lvm_read

m = "200us_100mW"

#SET PARAMETERS OF THE RUN
#laser pulse length (s)
pulse_length=200e-6

#time between each initialising pulses (will be the LIA half period)
init_pulse_separation = 30e-3

#so each probe happens each 30e-3 seconds.

#initial tau (difference in time)
init_tau = 240e-6
#final tau
fin_tau = 12.9e-3

#number of taus
n_tau = 50

taus = []
for i in range(n_tau):
    taus.append(init_tau + i * (fin_tau-init_tau)/n_tau)    
taus=np.array(taus)

if False:
    file_path = "T1_100mW_1.5s_pulse200us_2.lvm"
    data = np.genfromtxt(file_path, skip_header=23, delimiter=",")
    with open(m+'raw_data.pkl', 'wb') as k:
        pkl.dump(data,k)
else:
    with open(m+'raw_data.pkl','rb') as k:
        data=pkl.load(k)
        
datatrans=data.T

x = datatrans[0]
y = -datatrans[1]


####Here, we just calculate characteristics of the experiment from the input parameters.
#total time for one round of data
round_time = init_pulse_separation * 2 * n_tau
#numer of rounds
n_rounds = int(x[-1]/round_time)
#smallest_time between edges
smallest_diff = init_pulse_separation - pulse_length - fin_tau
#sampling rate
sample_length = x[1]-x[0]
#minimum number of points between edges
min_edge_sep = np.min(np.array([pulse_length, smallest_diff, init_tau-pulse_length]))/sample_length

fig, ax = plt.subplots(1,1)
ax.plot(x, y, label='Signal',color="deeppink")
ax.set_xlabel('Time')
ax.set_ylabel('Photocurrent')
ax.set_title('Signal vs Time')
fig.legend()
fig.tight_layout()
fig.savefig(m+'Initial_test.pdf')



#Find the edges of the pulses
#First, find the greatest difference
edges_list=[]
for i,v in enumerate(y[1:]):
    if np.abs(y[i+1]-y[i])>0.01:
        edges_list.append(i+1)
edges_arr = np.array(edges_list)

to_remove=[]
u = 0

while u==0:
    for i,edge in enumerate(edges_arr[:-1]):
        if edges_arr[i+1]-edges_arr[i]<min_edge_sep:
            to_remove.append(i)
            edges_arr = np.delete(edges_arr,i)
            u=0
            print('anomalous point found')
            break
        else:
            #print('not anomalous point')
            u=1
            continue


edges=edges_arr[:-1]

#odd_nums=np.arange(0,edges.shape[0],2)

fig, ax = plt.subplots(1,1)
ax.plot(x, y, label='Signal',color="deeppink")
ax.plot(x[edges], np.zeros_like(x[edges]), linestyle='None', marker="o", label='Edges',color="indigo")
ax.set_xlabel('Time')
ax.set_ylabel('Photocurrent')
ax.set_title('Signal vs Time')
fig.legend()
fig.tight_layout()
fig.savefig(m+'with_edges.pdf')

#Now add up all the data points for each, defined by the edge states
pulse_area_list=[]
odd_edges = edges[0::2]
even_edges= edges[1::2]

odd_nums=np.arange(0,edges.shape[0],2)
for i in odd_nums:
    area = np.sum(y[edges[i]:edges[i+1]])
    pulse_area_list.append(area)

pulse_area = np.array(pulse_area_list)

fig, ax = plt.subplots(1,1)
ax.plot(x[odd_edges], pulse_area, linestyle='None', marker="o", label='Edges',color="indigo")
ax.set_xlabel('Time')
ax.set_ylabel('Photocurrent')
ax.set_title('Signal vs Time')
fig.legend()
fig.tight_layout()
fig.savefig(m+'areas.pdf')

#Picking out the third one (the readout pulse)
readout_pos=np.arange(2,odd_edges.shape[0],3)
initial_pos=np.arange(1,odd_edges.shape[0],3)


fig, ax = plt.subplots(1,1)
ax.plot(x[odd_edges][readout_pos], pulse_area[readout_pos], linestyle='None', marker="o", label='Edges',color="indigo")
ax.set_xlabel('Time')
ax.set_ylabel('Photocurrent')
ax.set_title('Signal vs Time')
fig.legend()
fig.tight_layout()
fig.savefig(m+'readout_areas.pdf')

result_x = x[odd_edges][readout_pos]
result_y = pulse_area[readout_pos]
#find number of rounds done:
#n = result_y.shape/
offset = pulse_area[initial_pos]

# =============================================================================
# offset = []
# for i in odd_nums:
#     middle_int = int((edges[i]+edges[i+1])/2)
#     offset_1 = (edges[i+1]-edges[i]) * np.sum(y[middle_int:edges[i+1]])/(edges[i+1] - middle_int)
#     offset.append(offset_1)
# offset= np.array(offset)
# =============================================================================

fig, ax = plt.subplots(1,1)
ax.plot(result_x[:int((result_x.shape[0])/5)], -result_y[:int((result_x.shape[0])/5)], linestyle='None', marker="o", label='Edges',color="indigo")
ax.set_xlabel('Time')
ax.set_ylabel('Photocurrent')
ax.set_title('Signal vs Time')
fig.legend()
fig.tight_layout()
fig.savefig(m+'section.pdf')
#average_init = np.mean(np.array([offset[::int((result_x.shape[0])/5), ))
average = np.mean(np.array([result_y[:int((result_x.shape[0])/5)],result_y[int((result_x.shape[0])/5):int(2*(result_x.shape[0])/5)],result_y[int(2*(result_x.shape[0])/5):int(3*(result_x.shape[0])/5)],result_y[int(3*(result_x.shape[0])/5):int(4*(result_x.shape[0])/5)],result_y[int(4*(result_x.shape[0])/5):int(5*(result_x.shape[0])/5)]]),axis=0)
fig, ax = plt.subplots(1,1)
ax.plot(result_x[:int((result_x.shape[0])/5)], average, linestyle='None', marker="o", label='Edges',color="indigo")
ax.set_xlabel('Time')
ax.set_ylabel('Photocurrent')
ax.set_title('Signal vs Time')
ax.set_ylim(1,np.max(average)+0.5)
fig.legend()
fig.tight_layout()
fig.savefig('average_section.pdf')


fig, ax = plt.subplots(1,1)
ax.plot(result_x[:int((result_x.shape[0])/5)], average, linestyle='None', marker="o", label='Edges',color="indigo")
plt.yscale('log')
ax.set_xlabel('Time')
ax.set_ylabel('Photocurrent')
ax.set_ylim(1,np.max(average)+0.5)
ax.set_title('Signal vs Time')
fig.legend()
fig.tight_layout()
fig.savefig(m+'average_section_log.pdf')

#TRY ANOTHER WAY: only take the first quarter of the resonance
#Ok... now I see the whole pulse has changes involved. Instead, I'll take out the area of the first pulse as an offset. 

#Fitting with exponential decay
#average_initial = np.mean(np.array([offset[:int((offset.shape[0])/5)],offset[int((offset.shape[0])/5):int(2*(offset.shape[0])/5)],offset[int(2*(offset.shape[0])/5):int(3*(offset.shape[0])/5)],offset[int(3*(offset.shape[0])/5):int(4*(offset.shape[0])/5)],offset[int(4*(offset.shape[0])/5):int(5*(offset.shape[0])/5)]]),axis=0)
t=result_x[:int((result_x.shape[0])/5)]
first_run = result_y[:int((result_x.shape[0])/5)]-pulse_area[0]
average_init = np.mean(offset)
offset_av = average-average_init
fig, ax = plt.subplots(1,1)
ax.plot(t, offset_av, linestyle='None', marker="o", label='Edges',color="indigo")
ax.set_xlabel('Time')
ax.set_ylabel('Photocurrent')
ax.set_title('Signal vs Time')
fig.legend()
fig.tight_layout()
fig.savefig(m+'average_section_offset.pdf')

#just first round


fig, ax = plt.subplots(1,1)
ax.plot(t, first_run, linestyle='None', marker="o", label='Edges',color="indigo")
ax.set_xlabel('Time')
ax.set_ylabel('Photocurrent')
ax.set_title('Signal vs Time')
fig.legend()
fig.tight_layout()
fig.savefig(m+'average_section_offset_single_run.pdf')


#Fit with exponential
def decay(A, alpha, t, C):
    return A * np.exp(-alpha*t)
    
def lp(theta, t, y):
    A, alpha,C = theta
    residuals = y - decay(A, alpha, t,C)
    residual_variance = np.sum(residuals**2)
    return residual_variance

initial_guess = np.array([0.3,2,0.001])
args = (taus-100e-6, offset_av)


result = minimize(lp,
                     initial_guess,
                     args=args,
                     method="SLSQP",
                     bounds=[
                     (0, None),
                     (0, None),
                     (None, None),
                     ]
                     )

fitted_params=result.x

A_fit, alpha_fit,C_fit = fitted_params


###testing nelder-mead
result1 = minimize(lp,
                     initial_guess,
                     args=args,
                     method="nelder-mead",
                     bounds=[
                     (0, None),
                     (0, None),
                     (None, None),
                     ]
                     )

fitted_params1=result1.x

A_fit1, alpha_fit1,C_fit1 = fitted_params1

fig, ax = plt.subplots(1,1)
ax.plot(taus, offset_av, linestyle='None', marker="o", label='Data',color="indigo")
#ax.plot(taus,decay(A_fit, alpha_fit,taus),color="k",label='SLSQP Fit')
ax.plot(taus-100e-6,decay(A_fit1, alpha_fit1,taus-100e-6, C_fit1),color="red",label='Fit')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Photocurrent (V)')
#ax.set_title('Signal vs Time')
#ax.set_xlim(0,5)
fig.legend(loc = "upper right")
fig.tight_layout()
fig.savefig(m+'average_fit.pdf')

half_life = -(1/alpha_fit) * np.log(1/2)
print('Fitted parameters: A = ', A_fit1, ' alpha = ', alpha_fit1, ' C = ', C_fit1)
print('\n relaxation time: ', 1/alpha_fit1,' s.')


fig, ax = plt.subplots(1,1)
ax.annotate("",
            xy=(3.08995, 0.034), xycoords='data',
            xytext=(3.09045, 0.034), textcoords='data',
            arrowprops=dict(arrowstyle="-", color='blue', lw=1),
            )
ax.plot(x[odd_edges[154]-20:even_edges[155]+20], y[odd_edges[154]-20:even_edges[155]+20], label='Signal',color="deeppink")
ax.set_xlabel('Time (s)')
ax.set_ylabel('Photocurrent (V)')
#ax.set_title('Signal vs Time')
#fig.legend(loc="upper center")
fig.tight_layout()
fig.savefig(m+'Pulse_example.pdf')


if True:
    with open(m+'_T1.pkl', 'wb') as k:
        pkl.dump(1/alpha_fit1,k)
else:
    with open(m+'_T1.pkl','rb') as k:
        T1=pkl.load(k)