# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:29:21 2023

@author: imy1
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as mpatches
from scipy.optimize import check_grad

filename = 'PS5_data.csv'  

with open(filename, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # Read the header row
    x1=[]
    x2=[]
    c=[]
    X=[]
    # Process the data
    for row in reader:
        # Access data by column index
        x1_n=float(row[0])
        x2_n=float(row[1])
        c_n=float(row[2])
        X_n=np.vstack((x1_n,x2_n)).T[0]
        X.append(X_n)
        x1.append(x1_n)
        x2.append(x2_n)
        c.append(c_n)
X=np.array(X)
x1=np.array(x1)
x2=np.array(x2)
c=np.array(c)
n=x1.shape[0]

colours=['deeppink','indigo']
labels = ['c=1', 'c=0']

fig, axes = plt.subplots(1,1,figsize=(8,8))
axes.scatter(x1, x2, c=colours[0], alpha=c)
axes.scatter(x1, x2, c=colours[1], alpha=np.abs(c-1))
axes.set_ylabel("x2")
axes.set_xlabel("x1")
patches = [mpatches.Patch(color=colours[i], label=labels[i]) for i in range(2)]
axes.legend(handles=patches)
axes.set_title("Classifications of raw data")
fig.tight_layout
fig.savefig("coloured_plot.pdf")


############ QUESTION 1 AND 2 ##############

##We need to "normalise" the data so mean is 0 and standard deviation is 1.
x1_mean, x1_std = (np.mean(x1),np.std(x1))
x2_mean, x2_std = (np.mean(x2),np.std(x2))
X_mean, X_std = (np.mean(X,axis=0),np.std(X,axis=0))
x1new=(x1-x1_mean)/x1_std
x2new=(x2-x2_mean)/x2_std
Xnew=(X-X_mean)/X_std

#Defining training data with length t
t=int(x1.shape[0]*0.75)
X_t=Xnew[:t]
x1_t=x1new[:t]
x2_t=x2new[:t]
c_t=c[:t]

#Define testing data with length n-t
r=n-t
X_test=Xnew[:r]
x1_test=x1new[:r]
x2_test=x2new[:r]
c_test=c[:r]

#Plot the normalised data, just to see I've done it right.
fig, axes = plt.subplots(1,1,figsize=(8,8))
axes.scatter(x1new, x2new, c=colours[0], alpha=c)
axes.scatter(x1new, x2new, c=colours[1], alpha=np.abs(c-1))
axes.set_ylabel("x2")
axes.set_xlabel("x1")
patches = [mpatches.Patch(color=colours[i], label=labels[i]) for i in range(2)]
axes.legend(handles=patches)
axes.set_title("Classifications of normalised data")
fig.tight_layout
fig.savefig("normalised_plot.pdf")



################BUILDING NEURAL NETWORK################
#Define activation function
def activation(x):
    return 1 / (1 + np.exp(-np.clip(x,-15,15)))
    
#number of neurons in hidden layer
H=7

#Other parameters
num_epochs= 100000
losses = []
test_losses=[]
eta = 1e-3
np.random.seed(0)

#####Forward process
#weights for bias term:
w_b= np.random.randn(H)
#weights from x1 to 7 hidden neurons
w_x1=np.random.randn(H)
#weights from x2 to 7 hidden neurons
w_x2=np.random.randn(H)
#weights for hidden layer outputs to output neuron
w_out=np.random.randn(H+1)


for epoch in range(num_epochs):
    #HIDDEN LAYER
    hidden_layer_inputs = np.hstack([
        np.ones((t,1)),X_t])
    
    hidden_layer_weights = np.array([
        w_b, w_x1,w_x2
        ])
    
    alpha_h = hidden_layer_inputs @ hidden_layer_weights
    beta_h = activation(alpha_h).T
    
    #FIND TRAINING LOSS
    output_layer_inputs = np.hstack([
        np.ones((t,1)),beta_h.T])
    
    output_layer_weights = np.array([
        w_out
        ]).T
    
    output_layer_sums = output_layer_inputs @ output_layer_weights
    output_layer_outputs = activation(output_layer_sums)
    
    alpha_o=output_layer_sums
    beta_o = output_layer_outputs.T
    c_pred=beta_o.reshape(t,)
    eps=1e-15
    
    
    #Use binary cross-entropy function
    loss= -np.mean(c_t * np.log(c_pred + eps) + (1-c_t) * np.log(1-c_pred + eps))
    losses.append(loss)
    
    #DEFINE AND RECORD TEST LOSSES
    hidden_layer_inputs_test = np.hstack([np.ones((r, 1)), X_test])
    hidden_layer_weights_test = np.array([w_b, w_x1, w_x2])
    alpha_h_test = hidden_layer_inputs_test @ hidden_layer_weights_test
    beta_h_test = activation(alpha_h_test).T
    output_layer_inputs_test = np.hstack([
        np.ones((r,1)),beta_h_test.T])
    
    output_layer_weights_test = np.array([
        w_out
        ]).T
    
    output_layer_sums_test = output_layer_inputs_test @ output_layer_weights_test
    output_layer_outputs_test = activation(output_layer_sums_test)
    beta_o_test = output_layer_outputs_test.T
    c_pred_test=beta_o_test.reshape(-1)
    test_loss= -np.mean(c_test * np.log(c_pred_test + eps) + (1-c_test) * np.log(1-c_pred_test + eps))
    test_losses.append(test_loss)
    
    #Write the gradients
    #Look, I realise this would have been "easier" through matrix multiplication, but I didn't have
    #a good enough grasp of the patterns to do this. It was more straightforward for me to consider a case-by-case basis. Sorry.
    s = -c_t/c_pred + (1-c_t)/(1-c_pred)
    #Outer layer weights
    dw29 = s * c_pred*(1-c_pred) * beta_h[6]
    dw28 = s * c_pred*(1-c_pred) * beta_h[5]
    dw27 = s * c_pred*(1-c_pred) * beta_h[4]
    dw26 = s * c_pred*(1-c_pred) *beta_h[3]
    dw25 = s * c_pred*(1-c_pred) *beta_h[2]
    dw24 = s * c_pred*(1-c_pred) *beta_h[1]
    dw23 = s * c_pred*(1-c_pred) *beta_h[0]
    dw22 = s * c_pred*(1-c_pred)
    
    #Hidden layer weights (I did it by order of the hidden layers)
    dw1 = s * c_pred*(1-c_pred) * output_layer_weights[1] * beta_h[0] * (1-beta_h[0])* 1
    dw8 = s * c_pred*(1-c_pred) * output_layer_weights[1] * beta_h[0] * (1-beta_h[0])* x1_t
    dw15= s * c_pred*(1-c_pred) * output_layer_weights[1] * beta_h[0] * (1-beta_h[0])* x2_t
    dw2 = s * c_pred*(1-c_pred) * output_layer_weights[2] * beta_h[1] * (1-beta_h[1])* 1
    dw9 = s * c_pred*(1-c_pred) * output_layer_weights[2] * beta_h[1] * (1-beta_h[1])* x1_t
    dw16= s * c_pred*(1-c_pred) * output_layer_weights[2] * beta_h[1] * (1-beta_h[1])* x2_t
    dw3 = s * c_pred*(1-c_pred) * output_layer_weights[3] * beta_h[2] * (1-beta_h[2])* 1
    dw10= s * c_pred*(1-c_pred) * output_layer_weights[3] * beta_h[2] * (1-beta_h[2])* x1_t
    dw17= s * c_pred*(1-c_pred) * output_layer_weights[3] * beta_h[2] * (1-beta_h[2])* x2_t
    dw4 = s * c_pred*(1-c_pred) * output_layer_weights[4] * beta_h[3] * (1-beta_h[3])* 1
    dw11= s * c_pred*(1-c_pred) * output_layer_weights[4] * beta_h[3] * (1-beta_h[3])* x1_t
    dw18= s * c_pred*(1-c_pred) * output_layer_weights[4] * beta_h[3] * (1-beta_h[3])* x2_t
    dw5 = s * c_pred*(1-c_pred) * output_layer_weights[5] * beta_h[4] * (1-beta_h[4])* 1
    dw12= s * c_pred*(1-c_pred) * output_layer_weights[5] * beta_h[4] * (1-beta_h[4])* x1_t
    dw19= s * c_pred*(1-c_pred) * output_layer_weights[5] * beta_h[4] * (1-beta_h[4])* x2_t
    dw6 = s * c_pred*(1-c_pred) * output_layer_weights[6] * beta_h[5] * (1-beta_h[5])* 1
    dw13= s * c_pred*(1-c_pred) * output_layer_weights[6] * beta_h[5] * (1-beta_h[5])* x1_t
    dw20= s * c_pred*(1-c_pred) * output_layer_weights[6] * beta_h[5] * (1-beta_h[5])* x2_t
    dw7 = s * c_pred*(1-c_pred) * output_layer_weights[7] * beta_h[6] * (1-beta_h[6])* 1
    dw14= s * c_pred*(1-c_pred) * output_layer_weights[7] * beta_h[6] * (1-beta_h[6])* x1_t
    dw21= s * c_pred*(1-c_pred) * output_layer_weights[7] * beta_h[6] * (1-beta_h[6])* x2_t
   
    ##Allocate weights to different features
    gradientsb = np.array([dw1, dw2, dw3, dw4, dw5, dw6, dw7])
    gradientsx1 = np.array([dw8, dw9, dw10, dw11, dw12, dw13, dw14])
    gradientsx2 = np.array([dw15, dw16, dw17, dw18, dw19, dw20, dw21])
    gradientsout= np.array([dw22, dw23, dw24, dw25, dw26, dw27, dw28, dw29])
    
    #Update new weights
    new_w_b= w_b - eta * np.sum(gradientsb,axis=1)
    new_w_x1 = w_x1 - eta * np.sum(gradientsx1,axis=1)
    new_w_x2 = w_x2 - eta * np.sum(gradientsx2,axis=1)
    new_w_out = w_out - eta*np.sum(gradientsout,axis=1)
    w_b = new_w_b
    w_b = new_w_b 
    w_x1 = new_w_x1
    w_x2 = new_w_x2
    w_out = new_w_out

#Plot the losses
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(losses,label='Training loss')
ax.plot(test_losses,label='Test loss')
ax.set_xlabel(r"$Epoch$")
ax.set_ylabel(r"$Loss$")
fig.legend()
fig.tight_layout()
fig.savefig('loss.pdf')


#####CREATE MAP OF PREDICTIONS
def forward_pass(inputs,l):
    hidden_layer_inputs = np.hstack([np.ones((l, 1)), inputs])
    hidden_layer_weights = np.array([w_b, w_x1, w_x2])
    alpha_h = hidden_layer_inputs @ hidden_layer_weights
    beta_h = activation(alpha_h).T
    #OUTPUT LAYER
    output_layer_inputs = np.hstack([
        np.ones((l,1)),beta_h.T])
    
    output_layer_weights = np.array([
        w_out
        ]).T
    
    output_layer_sums = output_layer_inputs @ output_layer_weights
    output_layer_outputs = activation(output_layer_sums)
    beta_o = output_layer_outputs.T
    c_pred=beta_o.reshape(-1)
    return c_pred


###Set up points in plot
dim = 100
#Make map:
xx_min = -6.25
xx_max= -3.75
yy_min = -1000
yy_max=1250

xx1=np.linspace(xx_min, xx_max, dim)
xx2=np.linspace(yy_min, yy_max, dim)

XX=np.meshgrid(xx1,xx2)
XX_flat = np.column_stack([XX[0].ravel(), XX[1].ravel()])
#"Normalise" these points (reduce them with respect to the normalised training data)
XX_input=(XX_flat-X_mean)/X_std
points = XX_input.shape[0]
output=forward_pass(XX_input,points)
c_map=output.reshape((dim,dim))

#Plot the map!
fig, ax = plt.subplots(1,1)
contour = ax.contourf(xx1, xx2, c_map, cmap='coolwarm', alpha=0.6)
ax.scatter(x1, x2, c=colours[0], alpha=c)
ax.scatter(x1, x2, c=colours[1], alpha=np.abs(c-1))
ax.set_ylabel("x2")
ax.set_xlabel("x1")
ax.set_title('Smooth Map of Classifications')

# Add colorbar for the contour plot
cbar = plt.colorbar(contour)
cbar.set_label('Class')
# Step 8: Display the plot
fig.tight_layout()
fig.savefig('smooth_map2.pdf')



###############   QUESTION 3   #############
#REPEAT EVERYTHING
X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)

def gen_init(X, X_mean, X_std):
    x=(X-X_mean)/X_std
    return x

def init(X,c,X_mean, X_std):
    x=(X-X_mean)/X_std
    c=np.atleast_2d(c).T
    return x, c

t=75
r=100-t
x_total,y_total=init(X,c,X_mean,X_std)

#Define test and training data from this total
x=x_total[:t]
y=y_total[:t]
xt = x_total[-r:]
yt=y_total[-r:]

N, D_in = x.shape
N, D_out = y.shape
H = 7 # The number of neurons in the hidden layer.

x = torch.tensor(x, dtype=torch.float32)
xt=torch.tensor(xt, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
yt=torch.tensor(yt, dtype=torch.float32)

#Constructing the model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid(),
)
#Using BCE loss because c is binary
loss_fn = torch.nn.BCELoss(reduction = "sum")

#match the parameters in previous model
epochs = num_epochs
learning_rate = eta

losses = np.empty(epochs)
test_losses = np.empty(epochs)

for t in range(epochs):
    #Forward pass
    y_pred = model(x)

    #Compute losses for training and test
    loss = loss_fn(y_pred, y)
    losses[t] = loss.item()
    
    y_pred_test = model(xt)

    test_loss = loss_fn(y_pred_test, yt)
    test_losses[t]=test_loss.item()

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass.
    loss.backward()

    # Update the weights using gradient descent.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            

#Plot losses!
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(losses, label = 'Training loss')
ax.plot(test_losses,label='Test loss')
ax.set_xlabel(r"Epoch")
ax.set_ylabel(r"Loss")
fig.legend()
fig.tight_layout()
fig.savefig('PyTorch_Losses.pdf')

dim = 100

#Make map with appropriate bounds to match the unnormalised data
xx_min = -6.25
xx_max= -3.75
yy_min = -1000
yy_max=1250

xx1=np.linspace(xx_min, xx_max, dim)
xx2=np.linspace(yy_min, yy_max, dim)

XX=np.meshgrid(xx1,xx2)
XX_flat = np.column_stack([XX[0].ravel(), XX[1].ravel()])


XX_fn=gen_init(XX_flat,X_mean,X_std)

XX_input = torch.tensor(XX_fn,dtype=torch.float32)

c_map_results=model(XX_input)

c_map=c_map_results.detach().numpy().reshape(dim, dim)


#Plot the map!
fig, ax = plt.subplots(1,1)
contour = ax.contourf(xx1, xx2, c_map, cmap='coolwarm', alpha=0.6)
ax.scatter(x1, x2, c=colours[0], alpha=c)
ax.scatter(x1, x2, c=colours[1], alpha=np.abs(c-1))
ax.set_ylabel("x2")
ax.set_xlabel("x1")
ax.set_title('Smooth Map of Classifications')

cbar = plt.colorbar(contour)
cbar.set_label('Class')
# Step 8: Display the plot
fig.tight_layout()
fig.savefig('smooth_map3.pdf')
