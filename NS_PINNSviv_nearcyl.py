# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:10:41 2023

@author: 2919781
"""

"""
@author: Computational Domain
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import itertools

nu = 0.005

#%%
data = scipy.io.loadmat('udata_ur5_vivre200_subdefmesh2.mat') 

Udata_ur5_re200viv_orgsub2 = data['Udata_ur5_re200viv_orgsub2']  # N x T
Vdata_ur5_re200viv_orgsub2 = data['Vdata_ur5_re200viv_orgsub2']  # N x T
pdata_ur5_re200viv_orgsub2 = data['pdata_ur5_re200viv_orgsub2']  # N x T

Cxbase = data['Cxbase']
Cybase = data['Cybase'] 
  
 
origindisplacement_ur5_re200viv=data['origindisplacement_ur5_re200viv']

Nts=1000
spanindex=2
index=np.arange(0,Nts,spanindex)
Nsnap = len(index)
tmarray_span=index*0.02
 
 
nearcyl=np.where((Cxbase>-2)&(Cxbase<6)&(Cybase>-4)&(Cybase<4))
nearcyl_index=nearcyl[0]

Nxyn=len(nearcyl_index)

Udata_ur5_re200viv_orgsub_subdomain=Udata_ur5_re200viv_orgsub2[nearcyl_index,:]
Vdata_ur5_re200viv_orgsub_subdomain=Vdata_ur5_re200viv_orgsub2[nearcyl_index,:]
pdata_ur5_re200viv_orgsub_subdomain=pdata_ur5_re200viv_orgsub2[nearcyl_index,:]

u_array = Udata_ur5_re200viv_orgsub_subdomain[:,index].flatten()[:, None]
v_array = Vdata_ur5_re200viv_orgsub_subdomain[:,index].flatten()[:, None]
p_array = pdata_ur5_re200viv_orgsub_subdomain[:,index].flatten()[:, None] 
 
del Udata_ur5_re200viv_orgsub2 
del Vdata_ur5_re200viv_orgsub2 
del pdata_ur5_re200viv_orgsub2 

del Udata_ur5_re200viv_orgsub_subdomain
del Vdata_ur5_re200viv_orgsub_subdomain
del pdata_ur5_re200viv_orgsub_subdomain
del data

etax_read=origindisplacement_ur5_re200viv[index,0]
etay_read=origindisplacement_ur5_re200viv[index,1]

Ntt=len(index)

 

x2_array1 = np.tile( Cxbase[nearcyl_index].flatten()[:, None], (1,Ntt))
y2_array1 = np.tile( Cybase[nearcyl_index].flatten()[:, None], (1,Ntt)) 

x2_array = x2_array1.flatten()[:, None]
y2_array = y2_array1.flatten()[:, None] 

t_array1 = np.tile(tmarray_span.T, (1, Nxyn)).T  # N x T
t_array = t_array1.flatten()[:, None]
  

etax_array1 = np.tile(etax_read, (1, Nxyn)).T  # N x T
etax_array = etax_array1.flatten()[:, None]

etay_array1 = np.tile(etay_read, (1, Nxyn)).T  # N x T
etay_array = etay_array1.flatten()[:, None]


#%%


class NavierStokesVIVArchit_nearcyl():
    def __init__(self, X, Y, T,  u, v, p, eta_x, eta_y):

        self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(Y, dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(T, dtype=torch.float32, requires_grad=True)
         
        self.u = torch.tensor(u, dtype=torch.float32)
        self.v = torch.tensor(v, dtype=torch.float32)
        self.p = torch.tensor(p, dtype=torch.float32)

        self.eta_x = torch.tensor(eta_x, dtype=torch.float32)
        self.eta_y = torch.tensor(eta_y, dtype=torch.float32)

        #null vector to test against f and g:
        self.null = torch.zeros((self.x.shape[0], 1))

        # initialize network:
        self.network()

        self.optimizer = torch.optim.LBFGS(itertools.chain(self.netuvp.parameters(),self.neteta.parameters()), 
                                            max_iter=200000, max_eval=80000,
                                            history_size=50, tolerance_grad=1e-05, tolerance_change=0.5 * np.finfo(float).eps,
                                            line_search_fn="strong_wolfe")
   
        
        self.mse = nn.MSELoss()
         
        #loss
        self.ls = 0
        self.eqls = 0
        self.valls = 0

        #iteration number
        self.iter = 0

    def network(self):

        self.netuvp = nn.Sequential(
            nn.Linear(3, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 3))
        
        self.neteta = nn.Sequential(
            nn.Linear(1, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 2))

    def function(self, x, y, t):

        res1 = self.netuvp(torch.hstack((x, y, t)))
        u, v, p = res1[:, 0:1], res1[:, 1:2], res1[:, 2:3]         
        
        res2 = self.neteta(t)
        eta_x, eta_y = res2[:, 0:1], res2[:, 1:2] 
 
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        eta_x_t = torch.autograd.grad(eta_x, t, grad_outputs=torch.ones_like(eta_x), create_graph=True)[0]
        eta_x_tt = torch.autograd.grad(eta_x_t, t, grad_outputs=torch.ones_like(eta_x_t), create_graph=True)[0]

        eta_y_t = torch.autograd.grad(eta_y, t, grad_outputs=torch.ones_like(eta_y), create_graph=True)[0]
        eta_y_tt = torch.autograd.grad(eta_y_t, t, grad_outputs=torch.ones_like(eta_y_t), create_graph=True)[0]

        momentx = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy) + eta_x_tt
        momenty = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy) + eta_y_tt
        
        mass = u_x + v_y 

        return u, v, p, eta_x, eta_y, momentx, momenty, mass

    def closure(self):
        # reset gradients to zero:
        self.optimizer.zero_grad()

        # u, v, p, g and f predictions:
        u_prediction, v_prediction, p_prediction, eta_x_prediction, eta_y_prediction, momentx_prediction, momenty_prediction, mass_prediction = self.function(self.x, self.y, self.t)

        # calculate losses
        u_loss = self.mse(u_prediction, self.u)
        v_loss = self.mse(v_prediction, self.v)
        eta_x_loss = self.mse(eta_x_prediction, self.eta_x)
        eta_y_loss = self.mse(eta_y_prediction, self.eta_y)
        
        momentx_loss = self.mse(momentx_prediction, self.null)
        momenty_loss = self.mse(momenty_prediction, self.null)
        mass_loss = self.mse(mass_prediction, self.null)
        
        self.ls = u_loss + v_loss  + eta_x_loss + eta_y_loss + momentx_loss + momenty_loss + mass_loss
        self.eqls =  momentx_loss + momenty_loss + mass_loss
        self.valls = u_loss + v_loss + eta_x_loss + eta_y_loss

        # derivative with respect to net's weights:
        self.ls.backward()

        self.iter += 1
        if not self.iter % 1:
            print('{:} {:0.6f} {:0.6f} {:0.6f}'.format(self.iter, self.ls, self.eqls, self.valls))            
            
        return self.ls

    def train(self):

        # training loop
        self.netuvp.train()
        self.neteta.train()
        self.optimizer.step(self.closure)




#%%

N_train = 40000
  
# Training Data
idx = np.random.choice( Nxyn * Ntt, N_train, replace=False)
x_train = x2_array[idx, :]
y_train = y2_array[idx, :]
t_train = t_array[idx, :]
u_train = u_array[idx, :]
v_train = v_array[idx, :]
p_train = p_array[idx, :]
eta_x_train = etax_array[idx, :]
eta_y_train = etay_array[idx, :]

#%%
 
pinn = NavierStokesVIVArchit_nearcyl(x_train, y_train, t_train, u_train, v_train, p_train, eta_x_train, eta_y_train)

pinn.train()
 
  

modelname='modelvivur5nearcyl_deform_'+str(Nsnap)+'_'+str(N_train)+'.pt'

save_model = {
	'model_neteta': pinn.neteta.state_dict(),
	'model_netuvp': pinn.netuvp.state_dict(),
}
torch.save(save_model, modelname) 