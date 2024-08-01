import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pickle as pkl

# Specify the full path to your CSV file



#######################################################
if False:
    csv_file_path = 'C:/Users/imy1/Documents/Honours/Project/Research/ODMR-APD-needle_tip/first_1000ms-1MHz-2825MHz-2925MHz--0.20db.csv'
    datalist=[]
    # Step 1: Read the CSV file using the csv module
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip the header rows
        for _ in range(18):
            next(csv_reader)
        
        for row in csv_reader:
            floatrow = [float(entry) for entry in row[:3]]
            datalist.append(floatrow)
    datalist.pop()
    data=np.array(datalist)
    time1 = np.empty(data.shape[0])
    ch1_1 = np.empty(data.shape[0])
    ch2_1 = np.empty(data.shape[0])

    for i in range(data.shape[0]):
        time1[i]=data[i,0]+2825
        ch1_1[i]=data[i,1]
        ch2_1[i]=data[i,2]
    
    with open('time1.pkl','wb') as k:
        pkl.dump(time1,k)
    with open('ch1_1.pkl','wb') as k:
        pkl.dump(ch1_1,k)
    with open('ch2_1.pkl','wb') as k:
        pkl.dump(ch2_1,k)
        
        
    csv_file_path = 'C:/Users/imy1/Documents/Honours/Project/Research/ODMR-APD-needle_tip/1000ms-1MHz-2825MHz-2925MHz--0.20db-100pc.csv'
    datalist100pc=[]
    # Step 1: Read the CSV file using the csv module
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip the header rows
        for _ in range(18):
            next(csv_reader)
        
        for row in csv_reader:
            floatrow = [float(entry) for entry in row[:3]]
            datalist100pc.append(floatrow)
    datalist100pc.pop()
    data100pc=np.array(datalist100pc)
    time2 = np.empty(data100pc.shape[0])
    ch1_2 = np.empty(data100pc.shape[0])
    ch2_2 = np.empty(data100pc.shape[0])

    for i in range(data100pc.shape[0]):
        time2[i]=data100pc[i,0]+2825
        ch1_2[i]=data100pc[i,1]
        ch2_2[i]=data100pc[i,2]
    
    
    with open('time2.pkl','wb') as k:
        pkl.dump(time2,k)
    with open('ch1_2.pkl','wb') as k:
        pkl.dump(ch1_2,k)
    with open('ch2_2.pkl','wb') as k:
        pkl.dump(ch2_2,k)
        

    csv_file_path = 'C:/Users/imy1/Documents/Honours/Project/Research/ODMR-APD-needle_tip/1000ms-1MHz-2825MHz-2925MHz--0.20db-100pc-2.csv'
    datalist100pc2=[]
    # Step 1: Read the CSV file using the csv module
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip the header rows
        for _ in range(18):
            next(csv_reader)
        
        for row in csv_reader:
            floatrow = [float(entry) for entry in row[:3]]
            datalist100pc2.append(floatrow)
    datalist100pc2.pop()
    data100pc2=np.array(datalist100pc2)
    time3 = np.empty(data100pc2.shape[0])
    ch1_3 = np.empty(data100pc2.shape[0])
    ch2_3 = np.empty(data100pc2.shape[0])

    for i in range(data100pc2.shape[0]):
        time3[i]=data100pc2[i,0]+2825
        ch1_3[i]=data100pc2[i,1]
        ch2_3[i]=data100pc2[i,2]
    
    
    with open('time3.pkl','wb') as k:
        pkl.dump(time3,k)
    with open('ch1_3.pkl','wb') as k:
        pkl.dump(ch1_3,k)
    with open('ch2_3.pkl','wb') as k:
        pkl.dump(ch2_3,k)
        
    csv_file_path = 'C:/Users/imy1/Documents/Honours/Project/Research/ODMR-APD-needle_tip/direct_on_needle.csv'
    datalistdirect=[]
    # Step 1: Read the CSV file using the csv module
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip the header rows
        for _ in range(18):
            next(csv_reader)
        
        for row in csv_reader:
            floatrow = [float(entry) for entry in row[:3]]
            datalistdirect.append(floatrow)
    datalistdirect.pop()
    datadirect=np.array(datalistdirect)
    time4 = np.empty(datadirect.shape[0])
    ch1_4 = np.empty(datadirect.shape[0])
    ch2_4 = np.empty(datadirect.shape[0])

    for i in range(datadirect.shape[0]):
        time4[i]=datadirect[i,0]+2825
        ch1_4[i]=datadirect[i,1]
        ch2_4[i]=datadirect[i,2]
    
    
    with open('time4.pkl','wb') as k:
        pkl.dump(time4,k)
    with open('ch1_4.pkl','wb') as k:
        pkl.dump(ch1_4,k)
    with open('ch2_4.pkl','wb') as k:
        pkl.dump(ch2_4,k)   

    csv_file_path = 'C:/Users/imy1/Documents/Honours/Project/Research/ODMR-APD-needle_tip/long_lens.csv'
    datalistlong=[]
    # Step 1: Read the CSV file using the csv module
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip the header rows
        for _ in range(18):
            next(csv_reader)
        
        for row in csv_reader:
            floatrow = [float(entry) for entry in row[:3]]
            datalistlong.append(floatrow)
    datalistlong.pop()
    datalong=np.array(datalistlong)
    time5 = np.empty(datalong.shape[0])
    ch1_5 = np.empty(datalong.shape[0])
    ch2_5 = np.empty(datalong.shape[0])

    for i in range(datalong.shape[0]):
        time5[i]=datalong[i,0]+2825
        ch1_5[i]=datalong[i,1]
        ch2_5[i]=datalong[i,2]
    
    
    with open('time5.pkl','wb') as k:
        pkl.dump(time5,k)
    with open('ch1_5.pkl','wb') as k:
        pkl.dump(ch1_5,k)
    with open('ch2_5.pkl','wb') as k:
        pkl.dump(ch2_5,k)  

    csv_file_path = 'C:/Users/imy1/Documents/Honours/Project/Research/ODMR-APD-needle_tip/long_lens_2.csv'
    datalistlong2=[]
    # Step 1: Read the CSV file using the csv module
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip the header rows
        for _ in range(18):
            next(csv_reader)
        
        for row in csv_reader:
            floatrow = [float(entry) for entry in row[:3]]
            datalistlong2.append(floatrow)
    datalistlong2.pop()
    datalong2=np.array(datalistlong2)
    time6 = np.empty(datalong2.shape[0])
    ch1_6 = np.empty(datalong2.shape[0])
    ch2_6 = np.empty(datalong2.shape[0])

    for i in range(datalong2.shape[0]):
        time6[i]=datalong2[i,0]+2825
        ch1_6[i]=datalong2[i,1]
        ch2_6[i]=datalong2[i,2]
    
    
    with open('time6.pkl','wb') as k:
        pkl.dump(time6,k)
    with open('ch1_6.pkl','wb') as k:
        pkl.dump(ch1_6,k)
    with open('ch2_6.pkl','wb') as k:
        pkl.dump(ch2_6,k)   
        
    csv_file_path = 'C:/Users/imy1/Documents/Honours/Project/Research/ODMR-APD-needle_tip/through_needle.csv'
    datalistneedle=[]
    # Step 1: Read the CSV file using the csv module
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip the header rows
        for _ in range(18):
            next(csv_reader)
        
        for row in csv_reader:
            floatrow = [float(entry) for entry in row[:3]]
            datalistneedle.append(floatrow)
    datalistneedle.pop()
    dataneedle=np.array(datalistneedle)
    time7 = np.empty(dataneedle.shape[0])
    ch1_7 = np.empty(dataneedle.shape[0])
    ch2_7 = np.empty(dataneedle.shape[0])

    for i in range(dataneedle.shape[0]):
        time7[i]=dataneedle[i,0]+2825
        ch1_7[i]=dataneedle[i,1]
        ch2_7[i]=dataneedle[i,2]
    
    
    with open('time7.pkl','wb') as k:
        pkl.dump(time7,k)
    with open('ch1_7.pkl','wb') as k:
        pkl.dump(ch1_7,k)
    with open('ch2_7.pkl','wb') as k:
        pkl.dump(ch2_7,k) 

else:
    with open('time1.pkl','rb') as k:
        time1=pkl.load(k)
    with open('ch1_1.pkl','rb') as k:
        ch1_1=pkl.load(k)
    with open('ch2_1.pkl','rb') as k:
        ch2_1=pkl.load(k)
        
    with open('time2.pkl','rb') as k:
        time2=pkl.load(k)
    with open('ch1_2.pkl','rb') as k:
        ch1_2=pkl.load(k)
    with open('ch2_2.pkl','rb') as k:
        ch2_2=pkl.load(k)
        
    with open('time3.pkl','rb') as k:
        time3=pkl.load(k)
    with open('ch1_3.pkl','rb') as k:
        ch1_3=pkl.load(k)
    with open('ch2_3.pkl','rb') as k:
        ch2_3=pkl.load(k)
        
    with open('time4.pkl','rb') as k:
        time4=pkl.load(k)
    with open('ch1_4.pkl','rb') as k:
        ch1_4=pkl.load(k)
    with open('ch2_4.pkl','rb') as k:
        ch2_4=pkl.load(k)
        
    with open('time5.pkl','rb') as k:
        time5=pkl.load(k)
    with open('ch1_5.pkl','rb') as k:
        ch1_5=pkl.load(k)
    with open('ch2_5.pkl','rb') as k:
        ch2_5=pkl.load(k)
        
    with open('time6.pkl','rb') as k:
        time6=pkl.load(k)
    with open('ch1_6.pkl','rb') as k:
        ch1_6=pkl.load(k)
    with open('ch2_6.pkl','rb') as k:
        ch2_6=pkl.load(k)
        
    with open('time7.pkl','rb') as k:
        time7=pkl.load(k)
    with open('ch1_7.pkl','rb') as k:
        ch1_7=pkl.load(k)
    with open('ch2_7.pkl','rb') as k:
        ch2_7=pkl.load(k)
################################

r_1=np.sqrt(ch1_1**2 + ch2_1**2)
r_2=np.sqrt(ch1_2**2 + ch2_2**2)
r_3=np.sqrt(ch1_3**2 + ch2_3**2)
r_4=np.sqrt(ch1_4**2 + ch2_4**2)
r_5=np.sqrt(ch1_5**2 + ch2_5**2)
r_6=np.sqrt(ch1_6**2 + ch2_6**2)
r_7=np.sqrt(ch1_7**2 + ch2_7**2)


# Create plots
fig, ax = plt.subplots(1,1)
ax.plot(time1, r_1, label='Signal',color="deeppink")
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Signal vs MW frequency')
fig.legend()
fig.tight_layout()
fig.savefig('First.pdf')

fig, ax = plt.subplots(1,1)
ax.plot(time2, r_2, label='Signal',color="deeppink")
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Signal vs MW frequency')
fig.legend()
fig.tight_layout()
fig.savefig('Redo.pdf')

fig, ax = plt.subplots(1,1)
ax.plot(time3, r_3, label='Signal',color="deeppink")
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Signal vs MW frequency')
fig.legend()
fig.tight_layout()
fig.savefig('With_nearby_magnet.pdf')

fig, ax = plt.subplots(1,1)
ax.plot(time4, r_4, label='Signal',color="deeppink")
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Signal vs MW frequency')
fig.legend()
fig.tight_layout()
fig.savefig('Readjusted_direct.pdf')

fig, ax = plt.subplots(1,1)
ax.plot(time5, r_5, label='Signal',color="deeppink")
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Signal vs MW frequency')
fig.legend()
fig.tight_layout()
fig.savefig('long_lens.pdf')

fig, ax = plt.subplots(1,1)
ax.plot(time6, r_6, label='Signal',color="deeppink")
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Signal vs MW frequency')
ax.set_xlim([2860,2880])
fig.legend()
fig.tight_layout()
fig.savefig('long_lens_2.pdf')

fig, ax = plt.subplots(1,1)
ax.plot(time7, r_7, label='Signal',color="deeppink")
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Signal vs MW frequency')
fig.legend()
fig.tight_layout()
fig.savefig('through_needle.pdf')

"""
# Initialize empty lists to store data

datalist=[]
# Step 1: Read the CSV file using the csv module
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Skip the header rows
    for _ in range(18):
        next(csv_reader)
    
    for row in csv_reader:
        floatrow = [float(entry) for entry in row[:3]]
        datalist.append(floatrow)

datalist.pop()
data=np.array(datalist)
time = np.empty(data.shape[0])
ch1 = np.empty(data.shape[0])
ch2 = np.empty(data.shape[0])

for i in range(data.shape[0]):
    time[i]=data[i,0]+2825
    ch1[i]=data[i,1]
    ch2[i]=data[i,2]


# Create plots
fig, ax = plt.subplots(1,1)
ax.plot(time, ch1, label='CH1',color="deeppink")
ax.plot(time, ch2, label='CH2',color="lightseagreen")
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('CH1 and CH2 over Time')
fig.legend()
fig.tight_layout()

#    Fitting the data
#There are the following variables: 
#1) update log book
#2) look up transmitter stuff
#3) start planning thesis
#4: 11pm do patents (finish table)

def lorentzian(A, t0, gamma, t):
    return A * gamma**2 /((t-t0)**2 + gamma**2)
    
def ln_probability(theta, x, y):
    A1, t01, gamma1, A2, t02, gamma2 = theta
    residuals = y - lorentzian(A1, t01, gamma1, x) - lorentzian(A2, t02, gamma2, x) 
    residual_variance = np.var(residuals)
    ll = -0.5 * (np.sum(residuals**2 / residual_variance) + np.log(2*np.pi*residual_variance))
    return ll

x = time
y = ch1

initial_guess = [5, 2865, 10, 5, 2880, 10]
args = (x, y)

result = minimize(lambda *args: -ln_probability(*args),
                     initial_guess,
                     args=args,
                     method="Nelder-Mead")
fitted_params=result.x

A1m, t01m, gamma1m, A2m, t02m, gamma2m = fitted_params

# Create plots
fig, ax = plt.subplots(1,1)
ax.plot(x, y, label='CH1',color="deeppink")
ax.plot(x, ch2, label='CH2',color="lightseagreen")
ax.plot(x, lorentzian(A1m, t01m, gamma1m, x)+lorentzian(A2m, t02m, gamma2m, x), label='Fit',c="k")
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('CH1 and CH2 over Time (2)')
fig.legend()
fig.tight_layout()

print("Fitted Parameters:", fitted_params)


####### second one

csv_file_path_2 = 'C:/Users/imy1/Documents/Honours/Project/Research/ODMR-APD-needle_tip/1000ms-1MHz-2825MHz-2925MHz--0.20db-100pc.csv'

datalist_2=[]
# Step 1: Read the CSV file using the csv module
with open(csv_file_path_2, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Skip the header rows
    for _ in range(18):
        next(csv_reader)
    
    for row in csv_reader:
        floatrow = [float(entry) for entry in row[:3]]
        datalist_2.append(floatrow)

datalist_2.pop()
data_2=np.array(datalist_2)
time_2 = np.empty(data_2.shape[0])
ch1_2 = np.empty(data_2.shape[0])
ch2_2 = np.empty(data_2.shape[0])

for i in range(data_2.shape[0]):
    time_2[i]=data_2[i,0]+2825
    ch1_2[i]=data_2[i,1]
    ch2_2[i]=data_2[i,2]


# Create plots
fig, ax = plt.subplots(1,1)
ax.plot(time_2, ch1_2, label='CH1',color="deeppink")
ax.plot(time_2, ch2_2, label='CH2',color="lightseagreen")
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('CH1 and CH2 over Time (2)')
fig.legend()
fig.tight_layout()



csv_file_path_3 = 'C:/Users/imy1/Documents/Honours/Project/Research/ODMR-APD-needle_tip/1000ms-1MHz-2825MHz-2925MHz--0.20db-100pc-2.csv'

datalist_3=[]
# Step 1: Read the CSV file using the csv module
with open(csv_file_path_3, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Skip the header rows
    for _ in range(18):
        next(csv_reader)
    
    for row in csv_reader:
        floatrow = [float(entry) for entry in row[:3]]
        datalist_3.append(floatrow)

datalist_3.pop()
data_3=np.array(datalist_3)
time_3 = np.empty(data_3.shape[0])
ch1_3 = np.empty(data_3.shape[0])
ch2_3 = np.empty(data_3.shape[0])

for i in range(data_3.shape[0]):
    time_3[i]=data_3[i,0]+2825
    ch1_3[i]=data_3[i,1]
    ch2_3[i]=data_3[i,2]


# Create plots
fig, ax = plt.subplots(1,1)
ax.plot(time_3, ch1_3, label='CH1',color="deeppink")
ax.plot(time_3, ch2_3, label='CH2',color="lightseagreen")
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('CH1 and CH2 over Time with nearby magnet')
fig.legend()
fig.tight_layout()
"""