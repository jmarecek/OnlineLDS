# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# If you use this code, please cite our paper:
# @inproceedings{kozdoba2018,
#  title={On-Line Learning of Linear Dynamical Systems: Exponential Forgetting in Kalman Filters},
#  author={Kozdoba, Mark and Marecek, Jakub and Tchrakian, Tigran and Mannor, Shie},
#  booktitle = {The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19)},
#  note={arXiv preprint arXiv:1809.05870},
#  year={2019}
#}

# IBM-Review-Requirement: Art30.3
# Please note that the following code was developed for the project VaVeL at IBM Research 
# -- Ireland, funded by the European Union under the Horizon 2020 Program. 
# The project started on December 1st, 2015 and was completed by December 1st,
# 2018. Thus, in accordance with Article 30.3 of the Multi-Beneficiary General 
# Model Grant Agreement of the Program, the above limitations are in force.
# For further details please contact Jakub Marecek (jakub.marecek@ie.ibm.com), 
# or Gal Weiss (wgal@ie.ibm.com).

from __future__ import print_function
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from sklearn.metrics import f1_score
import time
import timeit
import math

import traceback

# Matlab loading
import tables
from scipy.io import loadmat

verbose = False

class hankel(object):
    def __init__(self,T):
        self.mat = np.matrix([[2./(pow(i+j+2,3)-(i+j+2)) for j in range(T)] for i in range(T)])
        [self.V, self.D] = np.linalg.eig(self.mat)

def wave_filtering_SISO(sys,T,k,eta,Rm):
   
    #Rm = 1/(1-max(np.linalg.eig(sys.A)[0]))

    n = sys.n
    m = sys.m

    k_dash = n*k + 2*n + m

    H = hankel(T)
    M = np.matrix(np.eye(m,k_dash))

    y_pred_full = []
    pred_error = []
    pred_error_persistent = []

    for t in range(1,T):
        X = []
        for j in range(k):
            scaling = pow(H.V[j], 0.25)
            conv = 0
            for u in range(0,t):
                conv += H.D[u,j] * sys.inputs[t-u]
            X.append(scaling * conv)
    
        X.append(sys.inputs[t-1])
        X.append(sys.inputs[t])
        X.append(sys.outputs[t-1])
        
        X = np.matrix(X).reshape(-1,1)
  
        y_pred = np.real(M*X)
        y_pred = y_pred[0,0] 
        y_pred_full.append(y_pred)
        #loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)
        loss = pow(np.linalg.norm(sys.outputs[t] + y_pred), 2)
        M = M - 2*eta*(sys.outputs[t] - y_pred)*X.transpose()
        frobenius_norm = np.linalg.norm(M,'fro')
        if frobenius_norm >= Rm:
            M = Rm/frobenius_norm * M

        pred_error.append(loss)
        pred_error_persistent.append(pow(np.linalg.norm(sys.outputs[t] -sys.outputs[t-1]), 2))

        #print(loss)

    return y_pred_full, M, pred_error, pred_error_persistent

def cost_ftl(M_flat, *args):
    
    n = args[0]
    m = args[1]
    T = args[2]
    Y = args[3]
    X = args[4]

    M = M_flat.reshape(n,m)
    '''
    J = 0
    for t in range(T):
        J += pow(np.linalg.norm(Y[:,t] - M*X[:,t]),2)
    '''

    J = np.real(np.trace(np.transpose(Y-M*X)*(Y-M*X)))
    
    return J

def gradient_ftl(M_flat, *args):

    n = args[0]
    m = args[1]
    T = args[2]
    Y = args[3]
    X = np.real(args[4])

    M = M_flat.reshape(n,m)
   
    '''
    dJ=np.matrix(np.zeros((n,m)))
    for t in range(T):
        dJ += M*X[:,t]*np.transpose(X[:,t]) - Y[:,t]*np.transpose(X[:,t])
   
    dJ *= 2
    '''

    dJ = 2*(M*X*X.transpose() - Y*X.transpose())

    return np.squeeze(np.array(dJ.reshape(-1,1)))


def wave_filtering_SISO_ftl(sys,T,k):
   
    n = sys.n
    m = sys.m

    k_dash = n*k + 2*n + m

    H = hankel(T)
    M = np.matrix(np.eye(m,k_dash))

    args4ftl = [0 for i in range(5)]
    args4ftl[0] = m
    args4ftl[1] = k_dash

    y_pred_full = []
    pred_error = []
    pred_error_persistent = []

    scalings = [pow(H.V[j], 0.25) for j in range(k)]
    for t in range(1,T):
        print_verbose("step %d of %d" % (t+1,T))
        X = []
        for j in range(k):
            scaling = scalings[j]
            conv = 0
            for u in range(t+1):
                conv += H.D[u,j] * sys.inputs[t-u]
            X.append(scaling * conv)
    
        X.append(sys.inputs[t-1])
        X.append(sys.inputs[t])
        X.append(sys.outputs[t-1])
        
        X = np.matrix(X).reshape(-1,1)
  
        y_pred = np.real(M*X)
        y_pred = y_pred[0,0] 
        y_pred_full.append(y_pred)
        loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)
        
        args4ftl[2] = t 
        
        try:
            args4ftl[3] = np.concatenate((args4ftl[3], sys.outputs[t]),1) 
            args4ftl[4] = np.concatenate((args4ftl[4], X),1)
        except:
            args4ftl[3] = sys.outputs[t]
            args4ftl[4] = X        
           
        args4ftl_tuple = tuple(i for i in args4ftl)  
 
        #result = opt.minimize(cost_ftl, M.reshape(-1,1), args=args4ftl_tuple, method='CG', jac=gradient_ftl)
        result = opt.minimize(cost_ftl, M.reshape(-1,1), args=args4ftl_tuple, jac=gradient_ftl)

        M = np.matrix(result.x).reshape(m,k_dash)
        pred_error.append(loss)
        pred_error_persistent.append(pow(np.linalg.norm(sys.outputs[t] -sys.outputs[t-1]), 2))


    return y_pred_full, M, pred_error, pred_error_persistent


#def do_filter_step(G,F,V,W, Id, Y_curr, m_prev,C_prev):
def Kalman_filtering_SISO(sys,T):

  G = np.diag(np.array(np.ones(4)))
  n = G.shape[0]

  F = np.ones(n)[:,np.newaxis] / np.sqrt(n)
  Id = np.eye(n)
  m_prev = 0
  C_prev = np.zeros((n,n))

  y_pred_full = [ 0 ]
  pred_error = [ sys.outputs[0] ]
    
  for t in range(1,T):    
    a = np.dot(G,m_prev)
    R = np.dot(G,np.dot(C_prev,G.T)) #+ W
    
    f = np.dot(F.T,a)    
    RF = np.dot(R,F)
    Q = np.dot(F.T,RF) #+ V
    A = RF
    try: A = RF / Q
    except: print("Zero Q? Check %s" % str(Q))
    
    #thats on purpose in a bit slower form, to test the equations
    y_pred = np.dot(F.T, np.dot(G,m_prev))
    m_prev = y_pred * A + np.dot((Id - np.dot(A,F.T)),a)
    C_prev = R - Q * np.dot(A,A.T)

    y_pred_full.append(y_pred)
    loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)
    pred_error.append(loss)
  
  return y_pred_full, pred_error

def cost_AR(theta, *args):
    '''
    theta: s parameters
    args[0]: observation at time t
    args[1]: past s observations (most most to least recent: t-1 to t-1-s)
    '''
    
    return pow(float(args[0]) - np.dot(args[1],theta),2)

def gradient_AR(theta, *args):
    '''
    theta: s parameters
    args[0]: observation
    args[1]: past s observations
    '''
    
    g = [(float(args[0]) - np.dot(args[1],theta) )  * i for i in args[1]]
    
    return np.squeeze(-2* np.array(g).reshape(-1,1))