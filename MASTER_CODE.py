#Below is the code for questions 1, 2, and 5. 

#=============================================================================
#======================== QUESTION 1 =========================================
#=============================================================================


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle as pkl
import george
import csv
from cmdstanpy import CmdStanModel
from george import kernels
from scipy.signal import lombscargle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import pandas as pd
from matplotlib.ticker import MaxNLocator
import emcee
import corner


filename = 'decay.csv'  
data=[]
with open(filename, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # Read the header row
    for row in reader:
        floatrow = [float(entry) for entry in row[:3]]
        if row[3] == 'A':
            det = np.array([1])
        if row[3] == 'B':
            det = np.array([2])
        if row[3] == 'C':
            det = np.array([3])
        finalrow=np.hstack((floatrow,det))
        data.append(finalrow)

data=np.array(data)


#======== PART 1 ============

def days2secs(days):
    secs = days* 24 * 60*60
    return secs

def secs2days(secs):
    days = secs/24/60/60
    return days

dataplot=data.T
#Data is seconds past, grams measured, uncertainty in grams measured, detector.
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.errorbar(secs2days(dataplot[0]), dataplot[1], yerr=dataplot[2], fmt="o", lw=2, c="k")
ax.set_xlabel("Days since manufacturing began")
ax.set_ylabel("Grams of radioactive material measured")
ax.set_title("Raw data")
fig.tight_layout
fig.savefig("raw.pdf")
        
#Formula is N=N0 exp(-alpha * (t-t0))      
#100 widgets  
Num_widgets = 100
#3 detectors
Num_detectors = 3

#initial radioactive mass N
N_0_max = 20  

#time taken to manufacture 
manufacturing_time = days2secs(35)
delay_time = days2secs(14)
investigation_time = days2secs(90)

#Number of observations per widget:
N_obs = 1

#Time measured
t_measured = dataplot[0]

N_measured = dataplot[1]

sigma_N_measured = dataplot[2]

detector = dataplot[3]
#number of points for each detector

NumA=np.where(detector==1)[0].shape[0]
NumB=np.where(detector==2)[0].shape[0]
NumC=np.where(detector==3)[0].shape[0]

nums=np.array([NumA,NumB,NumC])

#Lowest number of measurements in a detector
min_detect_num = min(NumA,NumB,NumC)

tA_measured = t_measured[np.where(detector==1)[0]]
tB_measured = t_measured[np.where(detector==2)[0]]
tC_measured = t_measured[np.where(detector==3)[0]]

tt_measured = []
tt_measured.append(tA_measured)
tt_measured.append(tB_measured)
tt_measured.append(tC_measured)
#np.column_stack([tA_measured,tB_measured,tC_measured]).T

NA_measured = N_measured[np.where(detector==1)[0]]
NB_measured = N_measured[np.where(detector==2)[0]]
NC_measured = N_measured[np.where(detector==3)[0]]

Nt_measured = []
Nt_measured.append(NA_measured)
Nt_measured.append(NB_measured)
Nt_measured.append(NC_measured)
#np.column_stack([NA_measured,NB_measured,NC_measured]).T

sA_measured = sigma_N_measured[np.where(detector==1)[0]]
sB_measured = sigma_N_measured[np.where(detector==2)[0]]
sC_measured = sigma_N_measured[np.where(detector==3)[0]]

sigma_Nt_measured = []
sigma_Nt_measured.append(sA_measured)
sigma_Nt_measured.append(sB_measured)
sigma_Nt_measured.append(sC_measured)
#np.column_stack([sA_measured,sB_measured,sC_measured]).T


#Use cmdstanpy to conduct inference.
ndata=data
iterations = 1000

if True:    
    ###SAMPLER
    
    start = {
    'alpha': 0.01,  # Provide a reasonable initial value for alpha
    'N_0': np.ones(Num_widgets),  # Provide initial values for N_0
    't_0': np.ones(Num_widgets) * 0.5 * (35*24*60*60),  # Provide initial values for t_0
    }
    
    line_model = CmdStanModel(stan_file='model.stan')
    
    line_data = {'Num_widgets': Num_widgets, 'N_measured': N_measured, 't_measured': t_measured,'sigma_N_measured': sigma_N_measured,'N_0_max': N_0_max}

    line_fit = line_model.sample(data=line_data, chains=2, iter_warmup=1000, iter_sampling=iterations)
    results = line_fit.stan_variables()
    
    with open('result.pkl','wb') as k:
        pkl.dump(results,k)
    with open('fit.pkl','wb') as k:
        pkl.dump(line_fit,k)
else:
    with open('result.pkl','rb') as k:
        results=pkl.load(k)
    with open('fit.pkl','rb') as k:
        line_fit=pkl.load(k)
        
#Save the chains for each parameter.
line_chains_t0 = results['t_0']
line_chains_N0 = results['N_0']
line_chains_alpha = results['alpha']

###Plot the chains 
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples = np.array(line_chains_alpha)     
label = (r'$\alpha$')
ax.plot(samples, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(0,0.000001)
fig.tight_layout()
plt.savefig('assignment_chain1_alpha.pdf') 

mean_alpha = np.mean(line_chains_alpha)
sigma_alpha = np.std(line_chains_alpha)

summary = line_fit.summary()

#Get the R-hat values
R_hat = summary['R_hat']
print(R_hat)
print(f"Mean value of inference for alpha is {mean_alpha}, with standard deviation {sigma_alpha} ")

#=========================== TRYING 3 BIASES =========================
if True:
    
    ###SAMPLER
    
    line_model = CmdStanModel(stan_file='biases_model.stan')
    for n in range(nums.shape[0]):
        line_data = {'Num_widgets': Num_widgets, 'N_0_max': N_0_max,'detector':detector,\
                     'N_measured': N_measured, 't_measured': t_measured,'sigma_N_measured': sigma_N_measured
                     }
    
        line_fit2 = line_model.sample(data=line_data, chains=2, iter_warmup=1000, iter_sampling=iterations)
        results2 = line_fit2.stan_variables()
    
    with open('result2.pkl','wb') as k:
        pkl.dump(results2,k)
    with open('fit2.pkl','wb') as k:
        pkl.dump(line_fit2,k)
else:
    with open('result2.pkl','rb') as k:
        results2=pkl.load(k)
    with open('fit2.pkl','rb') as k:
        line_fit2=pkl.load(k)


#Save the chains for each parameter.
line_chains_t0_1 = results2['t_0']
line_chains_N0_1 = results2['N_0']
line_chains_b1_1 = results2['b1']
line_chains_b2_1 = results2['b2']
line_chains_b3_1 = results2['b3']
line_chains_alpha_1 = results2['alpha']

###Plot the chains 
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples2 = np.array(line_chains_alpha_1)     
label = (r'$\alpha$')
ax.plot(samples2, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
ax.set_xlim(-100, iterations)
ax.set_ylim(0,0.000001)
fig.tight_layout()
plt.savefig('assignment_chain2_alpha.pdf') 


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samplesb = np.array(line_chains_b1_1)     
label = (r'$b_A$')
ax.plot(samplesb, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
ax.set_xlim(-100, iterations)
ax.set_ylim(-3,5)
fig.tight_layout()
plt.savefig('assignment_chain2_b1.pdf') 

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samplesb = np.array(line_chains_b2_1)     
label = (r'$b_B$')
ax.plot(samplesb, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
ax.set_xlim(-100, iterations)
ax.set_ylim(-3,5)
fig.tight_layout()
plt.savefig('assignment_chain2_b2.pdf') 

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samplesb = np.array(line_chains_b3_1)     
label = (r'$b_C$')
ax.plot(samplesb, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
ax.set_xlim(-100, iterations)
ax.set_ylim(-3,5)
fig.tight_layout()
plt.savefig('assignment_chain2_b3.pdf') 

mean_alpha_biases = np.mean(line_chains_alpha_1)
sigma_alpha_biases = np.std(line_chains_alpha_1)
mean_b1_biases = np.mean(line_chains_b1_1)
sigma_b1_biases = np.std(line_chains_b1_1)

mean_b2_biases = np.mean(line_chains_b2_1)
sigma_b2_biases = np.std(line_chains_b2_1)

mean_b3_biases = np.mean(line_chains_b3_1)
sigma_b3_biases = np.std(line_chains_b3_1)

summary1 = line_fit2.summary()

#Get the R-hat values
R_hat = summary1['R_hat']
print(f"The chain plots converge: \n mean R_hat = {R_hat}")
print(f"Biases: \n A: {mean_b1_biases} +/- {sigma_b1_biases} \n B: {mean_b2_biases} +/- {sigma_b2_biases} \n C: {mean_b3_biases} +/- {sigma_b3_biases}")
print(f"Alpha value with 3 biases is {mean_alpha_biases} +/- {sigma_alpha_biases}")


# ========= PART 2 ===========
#DO ANALYSIS ON ALL IN ONE STAN MODEL
line_chains_t0_2=np.zeros((3,iterations))
line_chains_N0_2=np.zeros((3,iterations))
line_chains_alpha_2=np.zeros(iterations)
line_chains_b_2=np.zeros(iterations)

if True:
       
   
    ###SAMPLER
    
    line_model = CmdStanModel(stan_file='all_model.stan')
    for n in range(nums.shape[0]):
        line_data = {'Num_widgets': Num_widgets, 'N_0_max': N_0_max,'detector':detector,\
                     'N_measured': N_measured, 't_measured': t_measured,'sigma_N_measured': sigma_N_measured
                     }
    
        line_fit3 = line_model.sample(data=line_data, chains=2, iter_warmup=1000, iter_sampling=iterations)
        results3 = line_fit3.stan_variables()
    
    with open('result3.pkl','wb') as k:
        pkl.dump(results3,k)
    with open('fit3.pkl','wb') as k:
        pkl.dump(line_fit3,k)
else:
    with open('result3.pkl','rb') as k:
        results3=pkl.load(k)
    with open('fit3.pkl','rb') as k:
        line_fit3=pkl.load(k)


#Save the chains for each parameter.
line_chains_t0_2 = results3['t_0']
line_chains_N0_2 = results3['N_0']
line_chains_b_2 = results3['b']
line_chains_alpha_2 = results3['alpha']

###Plot the chains 
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples2 = np.array(line_chains_alpha_2)     
label = (r'$\alpha$')
ax.plot(samples2, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(0,0.000001)
fig.tight_layout()
plt.savefig('assignment_chain3_alpha.pdf') 


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samplesb = np.array(line_chains_b_2)     
label = (r'$b$')
ax.plot(samplesb, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(-3,5)
fig.tight_layout()
plt.savefig('assignment_chain3_b.pdf') 

mean_alpha2 = np.mean(line_chains_alpha_2)
sigma_alpha2 = np.std(line_chains_alpha_2)
mean_b2 = np.mean(line_chains_b_2)
sigma_b2 = np.std(line_chains_b_2)

summary = line_fit3.summary()

# Get the R-hat values
R_hat = summary['R_hat']
print(f"The chain plots converge: \n R_hat = {R_hat}")
print(f"Values while incorporating bias parameter for detector B: \n mean bias = {mean_b2} +/- {sigma_b2} \n mean alpha = {mean_alpha2} +/- {sigma_alpha2}")






#=============================================================================
#======================== QUESTION 2 =========================================
#=============================================================================

import pickle
with open("assignment2_gp.pkl", "rb") as fp:
    data = pickle.load(fp)

t = data['t']
y = data['y']
yerr=data['yerr']


fig, ax = plt.subplots(1,1)
ax.errorbar(t,y,yerr=yerr,ecolor='gray',lw=2,linestyle='none',zorder=1,label="Uncertainty")
ax.scatter(t,y,c="k", s=2,zorder=10, label="Brightness")
ax.set_xlabel('Time (days)')
ax.set_ylabel('Brightness')
fig.legend()
fig.tight_layout()
fig.savefig('q2_raw.pdf')

#Using LombScargle
#define frequencies:
frequency = np.linspace(0.06, 270,5000)
power = lombscargle(t, y, frequency)
#finding the Lomb-Scargle periodogram

periods = (2*np.pi)/frequency

fig,ax=plt.subplots(1,1)
ax.plot(frequency, power)
ax.set_xlabel('Angular frequency')
ax.set_ylabel('Power')
ax.set_xlim(0.06,270)
fig.tight_layout()
fig.savefig('first_freq.pdf')

frequency = np.linspace(0.06, 10,5000)
power = lombscargle(t, y, frequency)
#finding the Lomb-Scargle periodogram

periods = (2*np.pi)/frequency

fig,ax=plt.subplots(1,1)
ax.plot(frequency, power)
ax.set_xlabel('Angular frequency')
ax.set_ylabel('Power')
ax.set_xlim(0.06,10)
fig.tight_layout()
fig.savefig('second_freq.pdf')

frequency = np.linspace(0.06, 5,5000)
power = lombscargle(t, y, frequency)

fig,ax=plt.subplots(1,1)
ax.plot(frequency, power)
ax.set_xlabel('Angular frequency')
ax.set_ylabel('Power')
ax.set_xlim(0.06,2)
fig.tight_layout()
fig.savefig('final_freq.pdf')

fig,ax=plt.subplots(1,1)
ax.plot(periods, power)
ax.set_xlabel('Period')
ax.set_ylabel('Power')
ax.set_xlim(0.02,100)
fig.tight_layout()
fig.savefig('periodogram.pdf')



#Using gaussian fitting to find the signal frequencies. 

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def four_gaussians(x, a1, x1, sigma1, a2, x2, sigma2, a3, x3, sigma3, a4, x4, sigma4):
    return (gaussian(x, a1, x1, sigma1) +
            gaussian(x, a2, x2, sigma2) +
            gaussian(x, a3, x3, sigma3) + 
            gaussian(x, a4, x4, sigma4))




# Initial guesses for the parameters
initial_guesses = [260, 0.12, 0.01, 1600,0.2, 0.01, 100, 0.35, 0.005, 900, 0.4, 0.01]
#initial_guesses = [900, (2*np.pi)/15, 0.1,100, (2*np.pi)/19,0.1, 1700, (2*np.pi)/29, 0.1,300, (2*np.pi)/52, 0.2]

# Perform the curve fit
popt, pcov = curve_fit(four_gaussians, frequency, power, p0=initial_guesses)

a1_fit, x1_fit, sigma1_fit,a2_fit, x2_fit, sigma2_fit,a3_fit, x3_fit, sigma3_fit,a4_fit, x4_fit, sigma4_fit = popt
a1_err, x1_err, sigma1_err,a2_err, x2_err, sigma2_err,a3_err, x3_err, sigma3_err,a4_err, x4_err, sigma4_err = np.sqrt(np.diag(pcov))

fig,ax = plt.subplots(1,1)
ax.scatter(frequency, power, label='Data', c="k", s=4)
#plt.plot(x, y_true, label='True Gaussian')
ax.plot(frequency, four_gaussians(frequency, *popt), label='Fit', c="deeppink")
ax.set_xlim(0.06,2)
ax.set_xlabel('Angular frequency')
ax.set_ylabel('Power')
fig.legend()
fig.tight_layout()
fig.savefig('fitted.pdf')

#Plot to find the relative strengths of the signals
norm_data = power  / (a2_fit)
fig,ax = plt.subplots(1,1)
ax.scatter(frequency, norm_data, label='Normalised power', c="k", s=4)
#plt.plot(x, y_true, label='True Gaussian')
ax.set_xlim(0.06,2)
ax.set_xlabel('Angular frequency')
ax.set_ylabel('Power')
fig.legend()
fig.tight_layout()
fig.savefig('normalised_fitted.pdf')
 

print(f" Fitted parameters for 3 Gaussians: \n First Gaussian: A = {a1_fit} +/- {a1_err}, mean = {x1_fit} +/- {x1_err}, std = {sigma1_fit} +/- {sigma1_err}\n \
      \n Second Gaussian: A = {a2_fit} +/- {a2_err}, mean = {x2_fit} +/- {x2_err}, std = {sigma2_fit} +/- {sigma2_err}\n  \
          \n Third Gaussian: A = {a3_fit} +/- {a3_err}, mean = {x3_fit} +/- {x3_err}, std = {sigma3_fit} +/- {sigma3_err}\n \
              \n Fourth Gaussian: A = {a4_fit} +/- {a4_err}, mean = {x4_fit} +/- {x4_err}, std = {sigma4_fit} +/- {sigma4_err}\n")

#============ PART 2 ============
#Gaussian process
#Create 4 kernels for 4 potential types of periodicity (just in case the fourth counts). 

# set data
time = t
brightness = y

# Define the kernel
#set gamma = small because effectively no decay
k1 = np.var(y) *0.2*kernels.ExpSine2Kernel(gamma=0.1, log_period=np.log((2*np.pi)/x1_fit),ndim=1) 
k2 = np.var(y)*1*kernels.ExpSine2Kernel(gamma=0.1, log_period=np.log((2*np.pi)/x2_fit),ndim=1)
k3 = np.var(y)*0.6*kernels.ExpSine2Kernel(gamma=0.1, log_period=np.log((2*np.pi)/x3_fit),ndim=1)
kernel = k3+k2+k1

gp = george.GP(kernel,mean=np.mean(y),fit_mean=True, white_noise=np.log(0.4**2), fit_white_noise=True)

#Create Gaussian process model
if True:
    gp.compute(t,yerr=yerr)
    print(gp.log_likelihood(y))
    print(gp.grad_log_likelihood(y))
    print(gp.kernel.parameter_names)
    
    def neg_log_likelihood(params):
        gp.set_parameter_vector(params)
        log_likelihood = -gp.log_likelihood(y,quiet=True)
        return log_likelihood if np.isfinite(log_likelihood) else 1e25
    
    def grad_nll(params):
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(y,quiet=True)
    
    #Compute the GP once:
    gp.compute(t, yerr=yerr)
    
    #print the initial log-likelihood:
    print(gp.log_likelihood(y))
    
    #Run optimization:
    p0=gp.get_parameter_vector()
    results = minimize(neg_log_likelihood, p0, jac=grad_nll,method="L-BFGS-B")
    
    #Update kernel, print final log-likelihood:
    gp.set_parameter_vector(results.x)
    print(gp.log_likelihood(y))
    
    # Generate time points for prediction
    future_t = np.linspace(max(t), 1360, 1000)
    mu, var = gp.predict(y,future_t,return_var=True)

    with open('results3_q2.pkl','wb') as k:
        pkl.dump(results,k)
    with open('mu3_q2.pkl','wb') as k:
        pkl.dump(mu,k)
    with open('var3_q2.pkl','wb') as k:
        pkl.dump(var,k)
else:
    with open('results3_q2.pkl','rb') as k:
        results=pkl.load(k)
    with open('mu3_q2.pkl','rb') as k:
        mu=pkl.load(k)
    with open('var3_q2.pkl','rb') as k:
        var=pkl.load(k)

#Regenerating for when we don't run the above:
future_t = np.linspace(max(t), 1360, 1000)
std=np.sqrt(var)

#Plot the prediction for the following 2 months (assumed to be 61 days)
fig,ax = plt.subplots(1,1)
ax.scatter(t,y,s=1,c="k",label="Data")
ax.fill_between(future_t, mu+std, mu-std, color='deeppink',alpha=0.5,label="Predicted")
ax.set_xlim(min(t),t[-1]+61)

ax.set_ylabel('Brightness')
ax.set_xlabel('Time')
fig.legend()
fig.tight_layout()
fig.savefig('prediction.pdf')

#=================USING EMCEE SAMPLING ==========


def lnprob(param):
    if np.any((-100>param[1:]) + (param[1:]>100)):
        return -np.inf
    
    gp.set_parameter_vector(param)
    return gp.lnlikelihood(y,quiet=True)

if True:
    gp.compute(t)
    
    #set up sampler
    nwalkers,ndim = 36, len(gp)
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob)
    
    #Initialise
    p0 = gp.get_parameter_vector() + 1e-4 * np.random.rand(nwalkers,ndim)
    
    print("Running burn-in")
    p0,_,_ = sampler.run_mcmc(p0,100,progress=True)
    
    print("running production chain")
    sampler.run_mcmc(p0,100,progress=True)
    
    fig, ax = plt.subplots(1,1)
    x = np.linspace(max(t),t[-1]+61,1000)
    
    stored=[]
    stored_chains=[]
    
    for i in range(50):
        w = np.random.randint(sampler.chain.shape[0])
        n = np.random.randint(sampler.chain.shape[1])
        gp.set_parameter_vector(sampler.chain[w,n])
        
        #plot single sample
        stored.append(gp.sample_conditional(y,x))
        stored_chains.append(sampler.chain)
        ax.plot(x,gp.sample_conditional(y,x),"deeppink",alpha=0.1)
    ax.scatter(t,y,c="k",s=1,label='Data')
    
    stored=np.array(stored)
    stored_chains=np.array(stored_chains)
    
    ax.set_xlim(t.min(),t[-1]+61)
    ax.set_xlabel(r"Days")
    ax.set_ylabel(r"Brightness")
    fig.tight_layout()
    fig.savefig('prediction_from_sampled_q2.pdf')
    
    with open('storedconditional_q2.pkl','wb') as k:
        pkl.dump(stored,k)
    with open('storedchains_q2.pkl','wb') as k:
        pkl.dump(stored_chains,k)
else:
    with open('storedconditional_q2.pkl','rb') as k:
        stored=pkl.load(k)
    with open('storedchains_q2.pkl','rb') as k:
        stored_chains=pkl.load(k)
        
    fig, ax = plt.subplots(1,1)
    x= np.linspace(max(t),t[-1]+61,1000)
    for i in range(stored.shape[0]):
        ax.plot(x,stored[i],"deeppink",alpha=0.1)
    ax.scatter(t,y,c="k",s=1,label='Data')
    ax.set_xlim(t.min(),t[-1]+61)
    ax.set_xlabel(r"Days")
    ax.set_ylabel(r"Brightness")
    fig.tight_layout()
    fig.savefig('other_prediction_sampled.pdf')
    




#=============================================================================
#======================== QUESTION 5 =========================================
#=============================================================================

df = pd.read_csv('unknown_data_cleaned.csv')
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

df=df.sort_values('Start',ascending=True).reset_index(drop=True)
df_dd = df.drop_duplicates()
df_dd_dna = df_dd.dropna()
print(df_dd_dna)



#================== BEFORE CLEANING ==================
#Calculate the difference from first measurement to find "starting time" and "ending time" spread in minutes
df['start_mins'] = (df['Start']-df['Start'][0]).dt.total_seconds() / 60

df['end_mins'] = (df['End']- df['Start'][0]).dt.total_seconds() / 60

# Calculate time difference in minutes between start and end times
df['time_diff'] = (df['End'] - df['Start']).dt.total_seconds() / 60

# Calculate time difference in minutes between end and start times
delay_times = []
for i in range(1, len(df['time_diff'])):
    delay_diff = ( df['Start'].iloc[i]- df['End'].iloc[i-1]).total_seconds() / 60
    delay_times.append(delay_diff)
delay_times = np.array(delay_times)

# Convert the 'time_diff' column to a NumPy array
time_diff_array = df['time_diff'].to_numpy()

# Print the time difference array
print(time_diff_array)





fig,ax=plt.subplots(1,1)
ax.scatter(range(len(df['end_mins'])),df['end_mins'],s=2)
ax.set_title('Plot of end times against experiment number')
fig.tight_layout()
fig.savefig('B4clean_et_steps.pdf')

fig,ax=plt.subplots(1,1)
ax.scatter(range(len(df['start_mins'])),df['start_mins'],s=2)
ax.set_title('Plot of start times against experiment number ')
fig.tight_layout()
fig.savefig('B4clean_st_steps.pdf')

fig,ax=plt.subplots(1,1)
ax.scatter(range(len(df['start_mins'])),df['start_mins'],s=2,label="Start time")
ax.scatter(range(len(df['end_mins'])),df['end_mins'],s=2,label="End time")
ax.set_title('Plot of start and end times against experiment number')
fig.tight_layout()
fig.savefig('B4clean_stet_together_steps.pdf')

fig,ax=plt.subplots(1,1)
ax.scatter(df['start_mins'],df['end_mins'],s=2,label="Start time")
fig.tight_layout()
fig.savefig('B4clean_stet_dependent_steps.pdf')

fig,ax=plt.subplots(1,1)
ax.scatter(range(len(time_diff_array)),time_diff_array,s=2)
fig.tight_layout()
fig.savefig('B4clean_time_diff_steps.pdf')

fig,ax=plt.subplots(1,1)
ax.scatter(range(len(delay_times)),delay_times,s=2)
fig.tight_layout()
fig.savefig('B4clean_delay_steps.pdf')


#================== AFTER CLEANING ===================
#Calculate the difference from first measurement to find "starting time" and "ending time" spread in minutes
df_dd_dna['start_mins'] = (df_dd_dna['Start']-df_dd_dna['Start'][0]).dt.total_seconds() / 60

df_dd_dna['end_mins'] = (df_dd_dna['End']- df_dd_dna['Start'][0]).dt.total_seconds() / 60

# Calculate time difference in minutes between start and end times
df_dd_dna['time_diff'] = (df_dd_dna['End'] - df_dd_dna['Start']).dt.total_seconds() / 60


#======= Find outliers for duration
#I will use an outlier mixture model for the durations to decide which values are outliers.
l = len(df_dd_dna['start_mins'])
y=df_dd_dna['time_diff']
x=np.array(range(l))


iterations = 2000
if True:    
    ###SAMPLER    
    
    line_model = CmdStanModel(stan_file='outlier.stan')
    
    line_data = {'N': 2, 'l': l, 'y': y, 'x': x}
    line_fit = line_model.sample(data=line_data, chains=2, iter_warmup=1000, iter_sampling=iterations)
    results = line_fit.stan_variables()
    
    with open('result_outlier.pkl','wb') as k:
        pkl.dump(results,k)
    with open('fit_outlier.pkl','wb') as k:
        pkl.dump(line_fit,k)
else:
    with open('result_outlier.pkl','rb') as k:
        results=pkl.load(k)
    with open('fit_outlier.pkl','rb') as k:
        line_fit=pkl.load(k)
        
line_chains_muR = results['mu_real']
line_chains_muO = results['mu_out']
line_chains_sigmaR = results['sigma_real']
line_chains_sigmaO = results['sigma_out']
line_chains_q = results['q']

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
label = [r"weights"]
ax.plot(line_chains_q, c="indigo", lw=1, label='Significant')
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(0,1)
fig.legend()
fig.tight_layout()
plt.savefig('q5_chain_w.pdf') 


####Retrieve probability estimates for the three models
log_prob1 = results["lp1"]
log_prob2 = results["lp2"]

#Define the log probability (including the weights).
q1 = np.zeros_like(log_prob1[0])
q2 = np.zeros_like(log_prob1[0])
for i in range(log_prob1.shape[0]):
    ll_R = np.log(line_chains_q[i]) + log_prob1[i]
    ll_O = np.log(1-line_chains_q[i]) + log_prob2[i]
    q1 += np.exp(ll_R - np.logaddexp(ll_R,ll_O))
    q2 += np.exp(ll_O - np.logaddexp(ll_R,ll_O))

#Normalise the relative values by dividing by number of samples.
q1 /= log_prob1.shape[0]
q2 /= log_prob1.shape[0]

#Combine into one matrix.
qmatrix=np.vstack((q1, q2))

fig,ax=plt.subplots(1,1)
ax.scatter(range(len(df_dd_dna['time_diff'])),df_dd_dna['time_diff'],s=5,c=q1, edgecolor="k", cmap="Greys")
ax.set_title('Plot of durations against experiment number')
fig.tight_layout()
fig.savefig('time_diff_outliers.pdf')

#Create threshold
threshold = 1e-8

print(f"SEE HERE: {q1.shape}, vs df: {df_dd_dna['time_diff'].shape}")

indices_dropped=np.where(q1<threshold)[0]
df_dd_dna=df_dd_dna.drop(df_dd_dna.index[indices_dropped])

fig,ax=plt.subplots(1,1)
ax.scatter(range(len(df_dd_dna['time_diff'])),df_dd_dna['time_diff'],s=2)
ax.set_title('Plot of durations against experiment number without outliers')
fig.tight_layout()
fig.savefig('B4clean_time_diff_steps_DROPPED.pdf')

# Calculate time difference in minutes between end and start times

indices_to_exclude=[]
for i in range(1, len(df_dd_dna)):
    delay_diff = ( df_dd_dna['Start'].iloc[i]- df_dd_dna['End'].iloc[i-1]).total_seconds() / 60
    if delay_diff<0:
        print('First set of indices to note:',i)
        indices_to_exclude.append(i)

df_dd_dna=df_dd_dna.drop(df_dd_dna.index[indices_to_exclude])


indices2=[]
for i in range(1, len(df_dd_dna['Start'])):
    delay_diff = ( df_dd_dna['Start'].iloc[i]- df_dd_dna['End'].iloc[i-1]).total_seconds() / 60
    if delay_diff<0:
        print('indices to note:',i)
        indices2.append(i)
df_dd_dna=df_dd_dna.drop(indices2)


delay_times = []
for i in range(1, len(df_dd_dna['Start'])):
    delay_diff = ( df_dd_dna['Start'].iloc[i]- df_dd_dna['End'].iloc[i-1]).total_seconds() / 60
    delay_times.append(delay_diff)
    if delay_diff<0:
        print('final indices to note:',i)
delay_times = np.array(delay_times)

# Convert the 'time_diff' column to a NumPy array
time_diff_array = df_dd_dna['time_diff'].to_numpy()




fig,ax=plt.subplots(1,1)
ax.scatter(range(len(df_dd_dna['end_mins'])),df_dd_dna['end_mins'],s=2)
ax.set_title('Plot of end times against experiment number')
fig.tight_layout()
fig.savefig('Clean_et_steps.pdf')

fig,ax=plt.subplots(1,1)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],s=2)
ax.set_title('Plot of start times against experiment number ')
fig.tight_layout()
fig.savefig('Clean_st_steps.pdf')

fig,ax=plt.subplots(1,1)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],s=2,label="Start time")
ax.scatter(range(len(df_dd_dna['end_mins'])),df_dd_dna['end_mins'],s=2,label="End time")
ax.set_title('Plot of start and end times against experiment number')
fig.tight_layout()
fig.savefig('Clean_stet_together_steps.pdf')

fig,ax=plt.subplots(1,1)
ax.scatter(df_dd_dna['start_mins'],df_dd_dna['end_mins'],s=2,label="Start time")
ax.set_title('Plot of end times against start times')
fig.tight_layout()
fig.savefig('Clean_stet_dependent_steps.pdf')

fig,ax=plt.subplots(1,1)
ax.scatter(range(len(df_dd_dna['time_diff'])),df_dd_dna['time_diff'],s=2)
ax.set_title('Plot of durations against experiment number')
fig.tight_layout()
fig.savefig('Clean_time_diff_steps.pdf')

fig,ax=plt.subplots(1,1)
ax.scatter(range(len(delay_times)),delay_times,s=2)
ax.set_title('Plot of time delays against experiment number')
fig.tight_layout()
fig.savefig('Clean_delay_steps.pdf')



#=======implement them into new data



#============ CONDUCTING INFERENCE =================
l = len(df_dd_dna['start_mins'])

def ln_prior(theta):
    c, m, sigma = theta
    return -3/2 * np.log(1 + m**2)
def ln_likelihood(theta, x, y):
    c, m, sigma = theta
    return -0.5 * np.sum((y - m * x - c)**2/sigma)

def ln_probability(theta, x, y):
    return ln_prior(theta) + ln_likelihood(theta, x, y)


#Initial guesses, using small error:
    

#Start vs end times
x1 = df_dd_dna['start_mins'].to_numpy()
y1=df_dd_dna['end_mins'].to_numpy()

x=x1
y=y1
y_err= np.ones_like(y1)

#LINEAR REGRESSION - NO UNCERTAINTIES

Y = np.atleast_2d(y).T

A = np.vstack([np.ones_like(x), x]).T
G = np.linalg.inv(A.T @ A)
X = G @ (A.T  @ Y)

c1_init = X[0][0]
m1_init = X[1][0]


x2 = np.array(range(l))
y2=df_dd_dna['start_mins'].to_numpy()
x=x2
y=y2
y_err= np.ones_like(y1)

Y = np.atleast_2d(y).T

A = np.vstack([np.ones_like(x), x]).T
G = np.linalg.inv(A.T @ A)
X = G @ (A.T  @ Y)

c2_init = X[0][0]
m2_init = X[1][0]

#======== USING CURVEFIT ========

def model(x, params):
    c, m, sigma = params
    y = np.random.normal(m*x + c, sigma)
    return y

def linear(x, c, m):
    return m*x + c



initial_guesses = [c1_init, m1_init]
# Perform the curve fit
popt1, pcov1 = curve_fit(linear, df_dd_dna['start_mins'], df_dd_dna['end_mins'], p0=initial_guesses)

c1_fit, m1_fit = popt1
c1_err, m1_err = np.sqrt(np.diag(pcov1))

initial_guesses = [c2_init, m2_init]
# Perform the curve fit
popt2, pcov2 = curve_fit(linear, np.array(range(l)), df_dd_dna['start_mins'], p0=initial_guesses)

c2_fit, m2_fit = popt2
c2_err, m2_err = np.sqrt(np.diag(pcov2))


print(f"end vs start: {c1_fit}+/-{c1_err} \n {m1_fit} +/- {m1_err}")
print(f"start vs experiment: {c2_fit}+/-{c2_err} \n {m2_fit} +/- {m2_err}")


fig, ax = plt.subplots(1,1)
x = np.linspace(l,l+1000,1000)
ax.plot(x, linear(x, c2_fit, m2_fit),c="indigo",alpha=1, label='Predicted start times',lw=1)
ax.plot(x, linear(linear(x, c2_fit, m2_fit), c1_fit, m1_fit),c="lightseagreen",alpha=1, label='Predicted end times',lw=1)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="k",s=1,label='Data')
ax.set_xlim(0,l+1000)
ax.set_xlabel(r"Experiment number")
ax.set_ylabel(r"Time")
fig.legend()
fig.tight_layout()
fig.savefig('prediction_q5.pdf')


#===============SAMPLING==========
#Note - I am plotting against time since the ORIGINAL start time.
#Start vs end times
x1 = df_dd_dna['start_mins'].to_numpy()
y1=df_dd_dna['end_mins'].to_numpy()

iterations = 2000
if True:
    x = x1
    y = y1
    
    start = {'sigma': np.std(y), 'c': c1_fit, 'm':m1_fit}
    
    line_model = CmdStanModel(stan_file='q5stan.stan')
    
    line_data = {'N': len(y),  'x': x1, 'y': y1}
    line_fit = line_model.sample(data=line_data, chains=2, iter_warmup=2000, iter_sampling=iterations, inits=start)
    results = line_fit.stan_variables()
    
    with open('resultq5_new.pkl','wb') as k:
        pkl.dump(results,k)
    with open('fitq5_new.pkl','wb') as k:
        pkl.dump(line_fit,k)
else:
    with open('resultq5_new.pkl','rb') as k:
        results=pkl.load(k)
    with open('fitq5_new.pkl','rb') as k:
        line_fit=pkl.load(k)

line_chains_c = results['c']
line_chains_sigma = results['sigma']
line_chains_m = results['m']


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples = np.array(line_chains_c)     
label = (r'$c$')
ax.plot(samples, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(-10,100)
fig.tight_layout()
plt.savefig('assignment_q5_chain_c.pdf') 

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples = np.array(line_chains_m)     
label = (r'$m$')
ax.plot(samples, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(0,2)
fig.tight_layout()
plt.savefig('assignment_q5_chain_m.pdf') 

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples = np.array(line_chains_sigma)     
label = (r'$\sigma$')
ax.plot(samples, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(0,100)
fig.tight_layout()
plt.savefig('assignment_q5_chain_sigma.pdf') 


mean_c = np.mean(line_chains_c)
sigma_c = np.std(line_chains_c)
mean_m = np.mean(line_chains_m)
sigma_m = np.std(line_chains_m)
mean_sigma = np.mean(line_chains_sigma)
sigma_sigma = np.std(line_chains_sigma)

summary = line_fit.summary()

# Get the R-hat values
R_hat = summary['R_hat']
print(R_hat)
print(f"Mean values: \n c: {mean_c} +/- {sigma_c} \n m: {mean_m} +/- {sigma_m} \n sigma_T: {mean_sigma} +/- {sigma_sigma}")






x2 = np.array(range(len(y1)))
y2=df_dd_dna['start_mins'].to_numpy()


if True:
    x = x2
    y = y2
    
    start = {'sigma': np.std(y), 'c': c2_fit, 'm':m2_fit}
    
    line_model = CmdStanModel(stan_file='q5stan2.stan')
    
    line_data = {'N': len(y),  'x': x2, 'y': y2}
    line_fit2 = line_model.sample(data=line_data, chains=2, iter_warmup=2000, iter_sampling=iterations,inits=start)
    results2 = line_fit2.stan_variables()
    
    with open('resultq5_2_new.pkl','wb') as k:
        pkl.dump(results2,k)
    with open('fitq5_2_new.pkl','wb') as k:
        pkl.dump(line_fit2,k)
else:
    with open('resultq5_2_new.pkl','rb') as k:
        results2=pkl.load(k)
    with open('fitq5_2_new.pkl','rb') as k:
        line_fit2=pkl.load(k)


line_chains_c2 = results2['c']
line_chains_sigma2 = results2['sigma']
line_chains_m2 = results2['m']


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples = np.array(line_chains_c2)     
label = (r'$c$')
ax.plot(samples, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(-50000,0)
fig.tight_layout()
plt.savefig('assignment_q5_chain_c2.pdf') 

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples = np.array(line_chains_m2)     
label = (r'$m$')
ax.plot(samples, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(0,400)
fig.tight_layout()
plt.savefig('assignment_q5_chain_m2.pdf') 

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples = np.array(line_chains_sigma2)     
label = (r'$\sigma$')
ax.plot(samples, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(0,15000)
fig.tight_layout()
plt.savefig('assignment_q5_chain_sigma2.pdf') 


mean_c2 = np.mean(line_chains_c2)
sigma_c2 = np.std(line_chains_c2)
mean_m2 = np.mean(line_chains_m2)
sigma_m2 = np.std(line_chains_m2)
mean_sigma2 = np.mean(line_chains_sigma2)
sigma_sigma2 = np.std(line_chains_sigma2)

summary2 = line_fit2.summary()

# Get the R-hat values
R_hat = summary2['R_hat']
print(R_hat)
print(f"Mean values: \n c: {mean_c2} +/- {sigma_c2} \n m: {mean_m2} +/- {sigma_m2} \n sigma_T: {mean_sigma2} +/- {sigma_sigma2}")



fig, ax = plt.subplots(1,1)
x = np.linspace(l,l+1000,1000)


for i in range(10):
    w = np.random.randint(line_chains_c2.shape[0])
    params2 = np.array([line_chains_c2[w], line_chains_m2[w],line_chains_sigma2[w]])
    ax.plot(range(int(x[-1])),model(range(int(x[-1])),params2),"deeppink",alpha=0.08)
    for j in range(10):
        k = np.random.randint(line_chains_c.shape[0])
        params = np.array([line_chains_c[k], line_chains_m[k],line_chains_sigma[k]])
        ax.plot(x,model(model(x,params2),params),"indigo",alpha=0.03)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="k",s=1,label='Measured start time',zorder=10)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="blue",s=1,label='Measured end time',zorder=9)

ax.set_xlim(0,l+1000)
ax.set_xlabel(r"Experiment number")
ax.set_ylabel(r"Time")
fig.legend()
fig.tight_layout()
fig.savefig('prediction_q5.pdf')



fig, ax = plt.subplots(1,1)
x = np.linspace(l,l+1000,1000)
for i in range(10):
    w = np.random.randint(line_chains_c2.shape[0])
    params2 = np.array([line_chains_c2[w], line_chains_m2[w],line_chains_sigma2[w]])
    ax.plot(range(int(x[-1])),linear(range(int(x[-1])),params2[0],params2[1]),"deeppink",alpha=0.08)
    for j in range(10):
        k = np.random.randint(line_chains_c.shape[0])
        params = np.array([line_chains_c[k], line_chains_m[k],line_chains_sigma[k]])
        ax.plot(x,linear(linear(x,params2[0],params2[1]),params[0],params[1]),"indigo",alpha=0.03)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="k",s=1,label='Measured start time',zorder=10)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="blue",s=1,label='Measured end time',zorder=9)



ax.set_xlim(0,l+1000)
ax.set_xlabel(r"Experiment number")
ax.set_ylabel(r"Time")
fig.legend()
fig.tight_layout()
fig.savefig('prediction_lines_q5.pdf')

mu2 = linear(x,mean_c2,mean_m2)
std2 = mean_sigma2
mu1 = linear(linear(x,mean_c2,mean_m2),mean_c,mean_m)
std1 = mean_sigma

fig, ax = plt.subplots(1,1)
x = np.linspace(l,l+1000,1000)
ax.plot(range(int(x[-1])),linear(range(int(x[-1])),mean_c2,mean_m2),"deeppink",alpha=1, label='Predicted start')
ax.plot(x,linear(linear(x,mean_c2,mean_m2),mean_c,mean_m),"indigo",alpha=0.03,label='Predicted end')
ax.fill_between(x, mu2+std2, mu2-std2, color='deeppink',alpha=0.3)
ax.fill_between(x, mu1+std1, mu1-std1, color='indigo',alpha=0.3)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="k",s=1,label='Measured start time',zorder=10)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="blue",s=1,label='Measured end time',zorder=9)
ax.set_xlim(0,l+1000)
ax.set_xlabel(r"Experiment number")
ax.set_ylabel(r"Time")
fig.legend()
fig.tight_layout()
fig.savefig('prediction_lines_q5.pdf')

######==========WHAT MODEL IS MOST APPROPRIATE?=========
#Checking with BIC

#I set the log likelihood of the system to be log of the LEAST SQUARES METHOD. 
def lnll(theta, x, y):
    t1, t2, t3, t4, t5 = theta
    residuals = y - t1 - t2*x - t3*x**2 - t4*x**3 - t5*x**4
    residual_variance = np.var(residuals)
    ll = -0.5 * (np.sum(residuals**2 / residual_variance) + y.shape[0]*np.log(2*np.pi*residual_variance))
    return np.sum(residuals)

#I set up the linear algebra (assuming no uncertainties, like given in the data)
#Model zero - flat
x= x2
y=y2


Y0 = np.atleast_2d(y).T

B0 = np.vstack([np.ones_like(x)]).T
G0 = np.linalg.inv(B0.T @ B0)
X0 = G0 @ (B0.T @ Y0)

#Model one - linear
Y1 = np.atleast_2d(y).T

B1 = np.vstack([np.ones_like(x), x]).T
G1 = np.linalg.inv(B1.T @ B1)
X1 = G1 @ (B1.T @ Y1)

#Model two - quadratic
Y2 = np.atleast_2d(y).T

B2 = np.vstack([np.ones_like(x), x, x**2]).T
G2 = np.linalg.inv(B2.T @ B2)
X2 = G2 @ (B2.T  @ Y2)

#Model fthree - cubic
Y3 = np.atleast_2d(y).T

B3 = np.vstack([np.ones_like(x), x, x**2, x**3]).T
G3 = np.linalg.inv(B3.T @ B3)
X3 = G3 @ (B3.T @ Y3)

#Model five - quartic
Y4 = np.atleast_2d(y).T

B4 = np.vstack([np.ones_like(x), x, x**2, x**3, x**4]).T
G4 = np.linalg.inv(B4.T @ B4)
X4 = G4 @ (B4.T @ Y4)

###Inserting into arrays of similar length so that we can define log likelihood:
xlist = np.zeros(5)

def BIC(X, x, y):
    N=y.shape[0]
    pos = list(range(0,np.size(X)))
    for i in pos:
        xlist[i] = X[i]
    D=np.size(X)
    logL=lnll(xlist, x, y)
    return (D*np.log(N) - (2 *logL))
BIC_const=BIC(X0, x, y)
BIC_lin = BIC(X1, x, y)
BIC_quad = BIC(X2, x, y)
BIC_cub = BIC(X3, x, y)
BIC_quart = BIC(X4,x, y)

BIC_plot=np.array([ BIC_const, BIC_lin, BIC_quad, BIC_cub,
                   BIC_quart])

x_plot = list(range(1,6))

fig,ax = plt.subplots(1,1)
ax.scatter(x_plot, BIC_plot, color='deeppink')
ax.xaxis.set_major_locator(MaxNLocator(8))
ax.yaxis.set_major_locator(MaxNLocator(8))
ax.set_title('Plot of BIC against number of parameters')
ax.set_xlabel('Number of parameters')
ax.set_ylabel('BIC')
plt.tight_layout()
plt.savefig('BIC_q5.pdf')
BICmin = np.min(BIC_plot)
xmin = x_plot[np.where(BIC_plot == BICmin)[0][0]]
print('Minimum point on BIC graph is (',xmin,BICmin,')')


#It seems like greater polynomial = better. 


def logfunc(theta, x, y):
    t1, t2, t3, t4, t5 = theta
    return (-0.5 * np.sum(np.log(y_err**2) + (y - t1 - t2*x - t3*x**2 - t4*x**3 - t5*x**4)**2 / y_err**2))

#Set some y_uncertainty. 
#I set up the linear algebra (assuming no uncertainties, like given in the data)
#Model zero - flat

y_err=y_err= np.ones_like(y1)*mean_sigma2
#Model one - flat line
Y1 = np.atleast_2d(y).T

B1 = np.vstack([np.ones_like(x)]).T
C1 = np.diag(y_err * y_err)

C1_inv = np.linalg.inv(C1)
G1 = np.linalg.inv(B1.T @ C1_inv @ B1)
X1 = G1 @ (B1.T @ C1_inv @ Y1)

#Model two - linear
Y2 = np.atleast_2d(y).T

B2 = np.vstack([np.ones_like(x), x]).T
C2 = np.diag(y_err * y_err)

C2_inv = np.linalg.inv(C2)
G2 = np.linalg.inv(B2.T @ C2_inv @ B2)
X2 = G2 @ (B2.T @ C2_inv @ Y2)

#Model three - quadratic
Y3 = np.atleast_2d(y).T

B3 = np.vstack([np.ones_like(x), x, x**2]).T
C3 = np.diag(y_err * y_err)

C3_inv = np.linalg.inv(C3)
G3 = np.linalg.inv(B3.T @ C3_inv @ B3)
X3 = G3 @ (B3.T @ C3_inv @ Y3)

#Model four - cubic
Y4 = np.atleast_2d(y).T

B4 = np.vstack([np.ones_like(x), x, x**2, x**3]).T
C4 = np.diag(y_err * y_err)

C4_inv = np.linalg.inv(C4)
G4 = np.linalg.inv(B4.T @ C4_inv @ B4)
X4 = G4 @ (B4.T @ C4_inv @ Y4)

#Model five - quartic
Y5 = np.atleast_2d(y).T

B5 = np.vstack([np.ones_like(x), x, x**2, x**3, x**4]).T
C5 = np.diag(y_err * y_err)

C5_inv = np.linalg.inv(C5)
G5 = np.linalg.inv(B5.T @ C5_inv @ B5)
X5 = G5 @ (B5.T @ C5_inv @ Y5)


#Model six - quintic
Y6 = np.atleast_2d(y).T

B6 = np.vstack([np.ones_like(x), x, x**2,x**3, x**4, x**5]).T
C6 = np.diag(y_err * y_err)

C6_inv = np.linalg.inv(C6)
G6 = np.linalg.inv(B6.T @ C6_inv @ B6)
X6 = G6 @ (B6.T @ C6_inv @ Y6)


###Inserting into arrays of similar length so that we can define log likelihood:
xlist = np.zeros(5)

def BIC(X, x, y):
    N=y.shape[0]
    pos = list(range(0,np.size(X)))
    for i in pos:
        xlist[i] = X[i]
    D=np.size(X)
    logL=logfunc(xlist, x, y)
    return (D*np.log(N) - (2 *logL))
BIC2_const=BIC(X1, x, y)
BIC2_lin = BIC(X2, x, y)
BIC2_quad = BIC(X3, x, y)
BIC2_cub = BIC(X4, x, y)
BIC2_quart = BIC(X5,x, y)

BIC_plot2=np.array([ BIC2_const, BIC2_lin, BIC2_quad, BIC2_cub,
                   BIC2_quart])

x_plot = list(range(1,6))

fig,ax = plt.subplots(1,1)
ax.scatter(x_plot, BIC_plot2, color='deeppink')
ax.xaxis.set_major_locator(MaxNLocator(8))
ax.yaxis.set_major_locator(MaxNLocator(8))
ax.set_title('Plot of BIC against number of parameters')
ax.set_xlabel('Number of parameters')
ax.set_ylabel('BIC')
plt.tight_layout()
plt.savefig('BIC_2_q5.pdf')
BICmin = np.min(BIC_plot2)
xmin = x_plot[np.where(BIC_plot2 == BICmin)[0][0]]
print('Minimum point on BIC graph is (',xmin,BICmin,')')


#REDOING EVERYTHING BUT FOR SQUARED RELATIONSHIP

#Initial guesses, using small error:
    
l = len(df_dd_dna['start_mins'])

#Start vs end times
x1 = df_dd_dna['start_mins'].to_numpy()
y1=df_dd_dna['end_mins'].to_numpy()

x=x1
y=y1
y_err= np.ones_like(y1)

#QUADRATIC REGRESSION - NO UNCERTAINTIES

Y = np.atleast_2d(y).T

A = np.vstack([np.ones_like(x), x,x**2]).T
G = np.linalg.inv(A.T @ A)
X = G @ (A.T  @ Y)

c1_init = X[0][0]
b1_init = X[1][0]
a1_init=X[2][0]


x2 = np.array(range(l))
y2=df_dd_dna['start_mins'].to_numpy()
x=x2
y=y2
y_err= np.ones_like(y1)

Y = np.atleast_2d(y).T

A = np.vstack([np.ones_like(x), x,x**2]).T
G = np.linalg.inv(A.T @ A)
X = G @ (A.T  @ Y)

c2_init = X[0][0]
b2_init = X[1][0]
a2_init = X[1][0]

#======== USING CURVEFIT ========

def quadmodel(x, params):
    c, b, a, sigma = params
    y = np.random.normal(a*x**2 +b*x + c, sigma)
    return y

def quadratic(x, c, b, a):
    return a*x*x + b*x + c




initial_guesses = [c2_init, b2_init, a1_init]
# Perform the curve fit
popt2, pcov2 = curve_fit(quadratic, np.array(range(l)), df_dd_dna['start_mins'], p0=initial_guesses)

c2_fit, b2_fit, a2_fit = popt2
c2_err, b2_err, a2_err = np.sqrt(np.diag(pcov2))


print(f"start vs experiment: {c2_fit}+/-{c2_err} \n {b2_fit} +/- {b2_err} \n {a2_fit} +/- {a2_err}")


fig, ax = plt.subplots(1,1)
x = np.linspace(l,l+1000,1000)
ax.plot(x, quadratic(x, c2_fit, b2_fit, a2_fit),c="indigo",alpha=1, label='Predicted start times',lw=1)
ax.plot(x, linear(quadratic(x, c2_fit, b2_fit,a2_fit), c1_fit, m1_fit),c="lightseagreen",alpha=1, label='Predicted end times',lw=1)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="k",s=1,label='Data')
ax.set_xlim(0,l+1000)
ax.set_xlabel(r"Experiment number")
ax.set_ylabel(r"Time")
fig.legend()
fig.tight_layout()
fig.savefig('prediction_quad_q5.pdf')


#===============SAMPLING==========
#****KEEP THE RESULTS FOR PREVIOUS SAMPLE, ONLY SAMPLE THE OTHER RELATIONSHIP******
#Note - I am plotting against time since the ORIGINAL start time.
#Start vs end times
x1 = df_dd_dna['start_mins'].to_numpy()
y1=df_dd_dna['end_mins'].to_numpy()

x2 = np.array(range(len(y1)))
y2=df_dd_dna['start_mins'].to_numpy()

iterations=1000
if True:
    x = x2
    y = y2
    
    start = {'c': c2_fit, 'b':b2_fit, 'a':a2_fit}
    
    line_model = CmdStanModel(stan_file='q5stan2_quad.stan')
    
    line_data = {'N': len(y),  'x': x2, 'y': y2}
    line_fit2 = line_model.sample(data=line_data, chains=2, iter_warmup=2000, iter_sampling=iterations)
    results2 = line_fit2.stan_variables()
    
    with open('resultq5_2_new_quad.pkl','wb') as k:
        pkl.dump(results2,k)
    with open('fitq5_2_new_quad.pkl','wb') as k:
        pkl.dump(line_fit2,k)
else:
    with open('resultq5_2_new_quad.pkl','rb') as k:
        results2=pkl.load(k)
    with open('fitq5_2_new_quad.pkl','rb') as k:
        line_fit2=pkl.load(k)


line_chains_c2_quad = results2['c']
line_chains_sigma2_quad = results2['sigma']
line_chains_b2_quad = results2['b']
line_chains_a2_quad = results2['a']


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples = np.array(line_chains_c2_quad)     
label = (r'$c$')
ax.plot(samples, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(0,10000)
fig.tight_layout()
plt.savefig('assignment_q5_chain_c2_quad.pdf') 

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples = np.array(line_chains_b2_quad)     
label = (r'$b$')
ax.plot(samples, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(0,200)
fig.tight_layout()
plt.savefig('assignment_q5_chain_b2_quad.pdf') 

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples = np.array(line_chains_a2_quad)     
label = (r'$a$')
ax.plot(samples, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(0,1)
fig.tight_layout()
plt.savefig('assignment_q5_chain_a2_quad.pdf') 

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
samples = np.array(line_chains_sigma2_quad)     
label = (r'$\sigma$')
ax.plot(samples, c="k", lw=1)
ax.set_xlabel(r"Step")
ax.set_ylabel(label)
# So we can see the initial behaviour:
ax.set_xlim(-100, iterations)
ax.set_ylim(0,5000)
fig.tight_layout()
plt.savefig('assignment_q5_chain_sigma2_quad.pdf') 


mean_c2_quad = np.mean(line_chains_c2_quad)
sigma_c2_quad = np.std(line_chains_c2_quad)
mean_b2_quad = np.mean(line_chains_b2_quad)
sigma_b2_quad = np.std(line_chains_b2_quad)
mean_a2_quad = np.mean(line_chains_a2_quad)
sigma_a2_quad = np.std(line_chains_a2_quad)
mean_sigma2_quad = np.mean(line_chains_sigma2_quad)
sigma_sigma2_quad = np.std(line_chains_sigma2_quad)

summary3 = line_fit2.summary()

# Get the R-hat values
R_hat = summary3['R_hat']
print(R_hat)
print(f"Mean values: \n c: {mean_c2_quad} +/- {sigma_c2_quad} \n b: {mean_b2_quad} +/- {sigma_b2_quad} \n a: {mean_a2_quad} +/- {sigma_a2_quad} \nsigma_T: {mean_sigma2_quad} +/- {sigma_sigma2_quad}")



fig, ax = plt.subplots(1,1)
x = np.linspace(l,l+1000,1000)


for i in range(10):
    w = np.random.randint(line_chains_c2_quad.shape[0])
    params2 = np.array([line_chains_c2_quad[w], line_chains_b2_quad[w],line_chains_a2_quad[w], line_chains_sigma2_quad[w]])
    ax.plot(range(int(x[-1])),quadmodel(np.array(range(int(x[-1]))),params2),"deeppink",alpha=0.08,zorder=8)
    for j in range(10):
        k = np.random.randint(line_chains_c.shape[0])
        params = np.array([line_chains_c[k], line_chains_m[k],line_chains_sigma[k]])
        ax.plot(x,model(quadmodel(x,params2),params),"indigo",alpha=0.03)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="k",s=1,label='Measured start time',zorder=10)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="blue",s=1,label='Measured end time',zorder=9)

ax.set_xlim(0,l+1000)
ax.set_xlabel(r"Experiment number")
ax.set_ylabel(r"Time")
fig.legend()
fig.tight_layout()
fig.savefig('prediction_q5_quad.pdf')



fig, ax = plt.subplots(1,1)
x = np.linspace(l,l+1000,1000)
for i in range(10):
    w = np.random.randint(line_chains_c2_quad.shape[0])
    params2 = np.array([line_chains_c2_quad[w], line_chains_b2_quad[w],line_chains_a2_quad[w], line_chains_sigma2_quad[w]])
    ax.plot(range(int(x[-1])),quadratic(range(int(x[-1])),params2[0],params2[1],params2[2]),"deeppink",alpha=0.08)
    for j in range(10):
        k = np.random.randint(line_chains_c.shape[0])
        params = np.array([line_chains_c[k], line_chains_m[k],line_chains_sigma[k]])
        ax.plot(x,linear(quadratic(x,params2[0],params2[1],params2[2]),params[0],params[1]),"indigo",alpha=0.03)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="k",s=1,label='Measured start time',zorder=10)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="blue",s=1,label='Measured end time',zorder=9)



ax.set_xlim(0,l+1000)
ax.set_xlabel(r"Experiment number")
ax.set_ylabel(r"Time")
fig.legend()
fig.tight_layout()
fig.savefig('prediction_lines_q5_quad.pdf')

mu2 = quadratic(x,mean_c2_quad,mean_b2_quad,mean_a2_quad)
std2 = mean_sigma2_quad
mu1 = linear(quadratic(x,mean_c2_quad,mean_b2_quad,mean_a2_quad),mean_c,mean_m)
std1 = mean_sigma

fig, ax = plt.subplots(1,1)
x = np.linspace(l,l+1000,1000)
ax.plot(range(int(x[-1])),quadratic(range(int(x[-1])),mean_c2_quad,mean_b2_quad, mean_a2_quad),"deeppink",alpha=1, label='Predicted start',zorder=8)
ax.plot(x,linear(quadratic(x,mean_c2_quad,mean_b2_quad,mean_a2_quad),mean_c,mean_m),"indigo",alpha=0.5,label='Predicted end')
ax.fill_between(x, mu2+std2, mu2-std2, color='deeppink',alpha=0.3,zorder=7)
ax.fill_between(x, mu1+std1, mu1-std1, color='indigo',alpha=0.3)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="k",s=1,label='Measured start time',zorder=10)
ax.scatter(range(len(df_dd_dna['start_mins'])),df_dd_dna['start_mins'],c="blue",s=1,label='Measured end time',zorder=9)
ax.set_xlim(0,l+1000)
ax.set_xlabel(r"Experiment number")
ax.set_ylabel(r"Time")
fig.legend()
fig.tight_layout()
fig.savefig('prediction_lines_q5_quad.pdf')


