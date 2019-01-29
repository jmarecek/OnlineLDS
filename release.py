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
# =============================================================================

# IBM-Review-Requirement: Art30.3
# Please note that the following code was developed for the project VaVeL 
# in DRL funded by the European Union under the Horizon 2020 Program. 
# The project started on December 1st, 2015 and was completed by December 1st,
# 2018. Thus, in accordance with Article 30.3 of the Multi-Beneficiary General 
# Model Grant Agreement of the Program, the above limitations are in force.
# For further details please contact the developers lead (jakub.marecek@ie.ibm.com), 
# or wgal@ie.ibm.com.


from __future__ import print_function
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import rlcompleter
from sklearn.metrics import f1_score
import time
import timeit
import math

# debugging
import pdb
pdb.Pdb.complete=rlcompleter.Completer(locals()).complete
import traceback

# Matlab loading
import tables
from scipy.io import loadmat


cost_count=0
grad_count=0

verbose = False

def print_verbose(a):
    if verbose: print(a)

def close_all_figs():
    plt.close('all')

class hankel(object):
    def __init__(self,T):
        self.mat = np.matrix([[2./(pow(i+j+2,3)-(i+j+2)) for j in range(T)] for i in range(T)])
        [self.V, self.D] = np.linalg.eig(self.mat)

class dynamical_system(object):
    def __init__(self,A,B,C,D, **kwargs):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        try:
            self.process_noise = kwargs['process_noise']
            try:
                self.proc_std = kwargs['process_noise_std']
            except KeyError:
                print('''Error: set 'process_noise_std'.''')
                exit()
        except KeyError:
            self.process_noise = None

        try:
            self.observation_noise = kwargs['observation_noise']
            try:
                self.obs_std = kwargs['observation_noise_std']
            except KeyError:
                print('''Error: set 'observation_noise_std'.''')
                exit()
           
        except KeyError:
            self.observation_noise = None
  
        # We expect to get a function that for a time-step T produces a multiplier
        # to be applied to b (possibly all elements of b, element-wise)
        try:
            self.timevarying_multiplier_b = kwargs['timevarying_multiplier_b']   
        except KeyError:
            self.timevarying_multiplier_b = None  
  
  
        # We expect to get a function that for a time-step T produces a multiplier
        # to be applied to b (possibly all elements of b, element-wise)
        try:
            self.corrupt_probability = kwargs['corrupt_probability']   
        except KeyError:
            self.corrupt_probability = None  
  
  
        #Checking dimensions of A and setting dimension, d, of state vector
        r = self.check_input(self.A)
        if r != 400:
            if r == 1:
                self.A=float(self.A)
                self.d=1
            else:
                self.A = np.matrix(self.A)
                if self.A.shape[0] != self.A.shape[1]:
                    print("Invalid state transition operator, A")
                    exit()
                self.d=self.A.shape[0]
        else:
            print("Invalid state transition operator, A")
            exit()

        #Checking dimensions of B and setting dimension, n, of input vector
        r = self.check_input(self.B)
        if r != 400:
            if r == 1:
                self.B=float(self.B)
                self.n=1
                if self.d != 1 and self.B !=0:
                    print("Invalid operator, B")
                    exit()
            else:
                self.B = np.matrix(self.B)
                if self.B.shape[0] != self.d:
                    print("Invalid operator, B")
                    exit()
                self.n=self.B.shape[1]
        else:
            print("Invalid operator, B")
            exit()

        #Checking dimensions of C and setting dimension, m, of observation vector
        r = self.check_input(self.C)
        if r != 400:
            if r == 1:
                self.C=float(self.C)
                self.m=1
                if self.d != 1:
                    print("Invalid operator, C")
                    exit()
            else:
                self.C = np.matrix(self.C)
                if self.C.shape[1] != self.d:
                    print("Invalid operator, C")
                    exit()
                self.m=self.C.shape[0]
        else:
            print("Invalid operator, C")
            exit()

        #Checking dimensions of D 
        r = self.check_input(self.D)
        if r != 400:
            if r == 1:
                self.D=float(self.D)
                if self.n != 1 and self.D != 0:
                    print("Invalid operator, D")
                    exit()
            else:
                self.D = np.matrix(self.D)
                if self.D.shape[1] != self.n:
                    print("Invalid operator, D")
                    exit()
        else:
            print("Invalid operator, D")
            exit()
 
    def check_input(self, operator):
        
        if isinstance(operator, int) or isinstance(operator, float):
            return 1
        else:
            try: 
                np.matrix(operator)
            except TypeError:
                return 400

    def solve(self, h0, inputs, T, **kwargs):
    
        if T == 1 or not isinstance(T,int): 
            print("T must be an integer greater than 1")
            exit()

        if self.d==1:
            try:
                h0=float(h0)
            except:
                print("Something wrong with initial state.")
                exit()
        else:
            try:
                h0 = np.matrix(h0, dtype=float).reshape(self.d,1)
            except:
                print("Something wrong with initial state.")
                exit()

        if self.n==1:
            try:
                self.inputs = list(np.squeeze(np.array(inputs, dtype=float).reshape(1,T)))
            except:
                print("Something wrong with inputs. Should be list of scalars of length %d." % (T))
                exit()
        else:
            try:
                self.inputs = np.matrix(inputs, dtype=float)
            except:
                print("Something wrong with inputs.")
                exit()

            if self.inputs.shape[0] != self.n or self.inputs.shape[1] !=T:
                print("Something wrong with inputs: wrong dimension or wrong number of inputs.")
                exit()

        if str(self.process_noise).lower() == 'gaussian':
            process_noise = np.matrix(np.random.normal(loc=0, scale=self.proc_std, size=(self.d,T)))
        else:
            process_noise = np.matrix(np.zeros((self.d,T)))

        if str(self.observation_noise).lower() == 'gaussian':
            observation_noise = np.matrix(np.random.normal(loc=0, scale=self.proc_std, size=(self.m,T)))
        else:
            observation_noise = np.matrix(np.zeros((self.m,T)))
        
        try:
            earliest_event_time = kwargs['earliest_event_time']
        except KeyError:
            earliest_event_time = 0

        self.h0=h0
        self.outputs = []
        self.event_or_not = []
        for t in range(T):        
            
            if self.n==1:
                h0 = self.A*h0 + self.B*self.inputs[t] + process_noise[:,t]
                y  = self.C*h0 + self.D*self.inputs[t] + observation_noise[:,t]
                if self.timevarying_multiplier_b is not None:
                    self.B *= self.timevarying_multiplier_b(t)
            else:
                h0 = self.A*h0 + self.B*self.inputs[:,t] + process_noise[:,t]
                y  = self.C*h0 + self.D*self.inputs[:,t] + observation_noise[:,t]  
                if self.timevarying_multiplier_b is not None:
                    self.B = self.B.dot(self.timevarying_multiplier_b(t))

            if (self.corrupt_probability is not None) and np.random.random_sample() <= self.corrupt_probability and t>earliest_event_time:
                self.event_or_not.append(True)
                y[:,0] = 100.0 * np.random.random_sample()
                self.outputs.append(y)
            else: 
                self.event_or_not.append(False)
                self.outputs.append(y)
        #print(self.outputs)




class time_series(object):
    def __init__(self, matlabfile = './OARIMA_code_data/data/setting6.mat', varname="seq_d0"):
        f = None
        self.outputs = []
        try: 
          f = tables.open_file(filename = matlabfile, mode='r')
          self.outputs = f.getNode('/' + varname)[:]
        except tables.exceptions.HDF5ExtError:
          print("Error in loading Matlab .dat from 7 upwards ... ")
          # print(traceback.format_exc())                     
        try: 
          if not f: 
              print("Loading Matlab .dat prior to version 7 instead.")
              print(loadmat(matlabfile).keys())
              self.outputs = list(loadmat(matlabfile)[varname][0])
        except:
          print("Error in loading Matlab .dat prior to version 7: ")
          print(traceback.format_exc())     
        self.m = 1
        print("Loaded %i elements in a series %s." % (len(self.outputs), varname))
        self.event_or_not = [False] * len(self.outputs)
        self.inputs = [0.0] * len(self.outputs) 
        self.h0 = 0
        self.n = 1

    def solve(self, h0 = [], inputs = [], T = 100, **kwargs):
        """ This just truncates the series loaded in the constructor."""
    
        if not isinstance(T,int): 
            print("T must be an integer. Anything less than 1 suggest no truncation is needed.")
            exit()
        print("Truncating to %i elements ..." % (T))
        self.h0=h0
        self.m = 1
        if T > 0:
          self.event_or_not = self.event_or_not[:T]
          self.inputs = self.inputs[:T]
          self.outputs = self.outputs[:T]

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
  
        #pdb.set_trace()
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
    
    global cost_count
    cost_count+=1
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

    global grad_count
    grad_count+=1
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


def testIdentification2(T = 100, noRuns = 10, sChoices = [15,3,1], haveKalman = False, haveSpectral = True, G = np.matrix([[0.999,0],[0,0.5]]), F_dash = np.matrix([[1,1]])):
  if haveKalman: sChoices = sChoices + [T]
      
  if noRuns < 2:
    print("Number of runs has to be larger than 1.")
    exit()

  filename = 'AR.pdf'
  pp = PdfPages(filename)

  ################# SYSTEM ###################        
  proc_noise_std = 0.5
  obs_noise_std  = 0.5
  
  error_spec_data = None
  error_persist_data = None
  error_AR1_data = None
  error_Kalman_data = None
  
  for runNo in range(noRuns):  
    sys = dynamical_system(G,np.zeros((2,1)),F_dash,np.zeros((1,1)),
      process_noise='gaussian',
      observation_noise='gaussian', 
      process_noise_std=proc_noise_std, 
      observation_noise_std=obs_noise_std,
      timevarying_multiplier_b = None)
    inputs = np.zeros(T)
    sys.solve([[1],[1]],inputs,T)
    Y = [i[0,0] for i in sys.outputs]
    #pdb.set_trace()
    ############################################
  
    ########## PRE-COMPUTE FILTER PARAMS ###################
    n = G.shape[0]
    m = F_dash.shape[0]

    W = proc_noise_std**2 * np.matrix(np.eye(n))
    V = obs_noise_std**2 * np.matrix(np.eye(m))
 
    #m_t = [np.matrix([[0],[0]])] 
    C = [np.matrix(np.eye(2))]
    R = []
    Q = []
    A = [] 
    Z = []

    for t in range(T):
        R.append(G * C[-1] * G.transpose() + W)
        Q.append(F_dash * R[-1] * F_dash.transpose() + V)
        A.append(R[-1]*F_dash.transpose()*np.linalg.inv(Q[-1]))
        C.append(R[-1] - A[-1]*Q[-1]*A[-1].transpose() )
        Z.append(G*( np.eye(2) - A[-1] * F_dash ))

    #PREDICTION
    plt.plot(Y, label='Output', color='#000000', linewidth=2, antialiased = True)
    for s in sChoices:
        Y_pred=[]
        for t in range(T):
            Y_pred_term1 = F_dash * G * A[t] * sys.outputs[t]

            if t==0:
                Y_pred.append(Y_pred_term1)
                continue

            acc = 0
            for j in range(min(t,s)+1):
                for i in range(j+1):
                    if i==0: 
                        ZZ=Z[t-i]
                        continue

                    ZZ = ZZ*Z[t-i]

                acc += ZZ * G * A[t-j-1] * Y[t-j-1]
                
            Y_pred.append(Y_pred_term1 + F_dash*acc)
    
        #print(np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(len(Y))]))

        #print(lab)
        if s == 1:
            if error_AR1_data is None: error_AR1_data = np.array([pow(np.linalg.norm(Y_pred[i][0,0] - Y[i]), 2) for i in range(len(Y))])
            else: 
                #print(error_AR1_data.shape)
                error_AR1_data = np.vstack((error_AR1_data, [pow(np.linalg.norm(Y_pred[i][0,0] - Y[i]), 2) for i in range(len(Y))]))
        if s == T:
            # For the spectral filtering etc, we use: loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)
            if error_Kalman_data is None: error_Kalman_data = np.array([pow(np.linalg.norm(Y_pred[i][0,0] - Y[i]), 2) for i in range(len(Y))])
            else: error_Kalman_data = np.vstack((error_Kalman_data, [pow(np.linalg.norm(Y_pred[i][0,0] - Y[i]), 2) for i in range(len(Y))]))
            plt.plot([i[0,0] for i in Y_pred], label="Kalman", color=(42.0/255.0, 204.0 / 255.0, 200.0/255.0), linewidth=2, antialiased = True)
        else:
            plt.plot([i[0,0] for i in Y_pred], label='AR(%i)' % (s+1), color=(42.0/255.0, 204.0 / 255.0, float(min(255.0,s))/255.0), linewidth=2, antialiased = True)
        
        plt.xlabel('Time')
        plt.ylabel('Prediction') 
        
    
    if haveSpectral:    
      predicted_output, M, error_spec, error_persist = wave_filtering_SISO_ftl(sys, T, 5)
      plt.plot(predicted_output, label='Spectral', color='#1B2ACC', linewidth=2, antialiased = True)
      if error_spec_data is None: error_spec_data = error_spec
      else: error_spec_data = np.vstack((error_spec_data, error_spec))
      if error_persist_data is None: error_persist_data = error_persist
      else: error_persist_data = np.vstack((error_persist_data, error_persist))
    
    plt.legend()
    plt.savefig(pp, format='pdf') 
    plt.close('all')  
    #plt.show()

  if haveSpectral:
    error_spec_mean = np.mean(error_spec_data, 0)
    error_spec_std = np.std(error_spec_data, 0)
    error_persist_mean = np.mean(error_persist_data, 0)
    error_persist_std = np.std(error_persist_data, 0)    

  error_AR1_mean = np.mean(error_AR1_data, 0)
  error_AR1_std = np.std(error_AR1_data, 0)    
  if haveKalman:
    error_Kalman_mean = np.mean(error_Kalman_data, 0)
    error_Kalman_std = np.std(error_Kalman_data, 0)    

  for (ylim, alphaValue) in [((0, 100.0), 0.2), ((0.0, 1.0), 0.05)]:
   for Tlim in [T-1, min(T-1, 20)]:
    #p3 = plt.figure()
    p3, ax = plt.subplots()
    plt.ylim(ylim)
    if haveSpectral:
      plt.plot(range(0,Tlim), error_spec[:Tlim], label='Spectral', color='#1B2ACC', linewidth=2, antialiased = True)
      plt.fill_between(range(0,Tlim), (error_spec_mean-error_spec_std)[:Tlim], (error_spec_mean+error_spec_std)[:Tlim], alpha=alphaValue, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=1, antialiased=True)
      plt.plot(range(0,Tlim), error_persist[:Tlim], label='Persistence', color='#CC1B2A', linewidth=2, antialiased = True)
      plt.fill_between(range(0,Tlim), (error_persist_mean-error_persist_std)[:Tlim], (error_persist_mean+error_persist_std)[:Tlim], alpha=alphaValue, edgecolor='#CC1B2A', facecolor='#FF0800', linewidth=1, antialiased=True)
    #import matplotlib.transforms as mtransforms
    #trans = mtransforms.blended_transform_factory(ax.transData, ax.transData)
    #trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    cAR1 = (42.0/255, 204.0 / 255.0, 1.0/255)
    bAR1 = (1.0, 204.0 / 255.0, 0.0) # , alphaValue
    print(cAR1)
    print(bAR1)
    #print(error_AR1_data)
    #print(error_AR1_mean)
    #print(Tlim)
    plt.plot(error_AR1_mean[:Tlim], label='AR(2)', color=cAR1, linewidth=2, antialiased = True)
    plt.fill_between(range(0,Tlim), (error_AR1_mean-error_AR1_std)[:Tlim], (error_AR1_mean+error_AR1_std)[:Tlim], alpha=alphaValue, edgecolor=cAR1, facecolor=bAR1, linewidth=1, antialiased=True) #transform=trans) #offset_position="data") alpha=alphaValue, 
    if haveKalman:
      cK = (42.0/255.0, 204.0 / 255.0, 200.0/255.0)
      bK = (1.0, 204.0 / 255.0, 200.0/255.0) # alphaValue
      print(cK)
      print(bK)
      plt.plot(error_Kalman_mean[:Tlim], label='Kalman', color=cK, linewidth=2, antialiased = True)
      plt.fill_between(range(0,Tlim), (error_Kalman_mean-error_Kalman_std)[:Tlim], (error_Kalman_mean+error_Kalman_std)[:Tlim], alpha=alphaValue, facecolor=bK, edgecolor=cK, linewidth=1, antialiased=True) # transform = trans) #offset_position="data") 
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Error')    
    #p3.show()
    p3.savefig(pp, format='pdf')
    
  pp.close()


# This is taken from pyplot documentation
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def testNoiseImpact(T = 50, noRuns = 10, discretisation = 10):
 filename = 'noise.pdf'
 pp = PdfPages(filename)
 
 for s in [1, 2, 3, 7]:
  data = np.zeros((discretisation, discretisation))
  diff = np.zeros((discretisation, discretisation))
  ratio = np.zeros((discretisation, discretisation))
  errKalman = np.zeros((discretisation, discretisation))
  errAR = np.zeros((discretisation, discretisation))
  ################# SYSTEM ###################        
  G = np.matrix([[0.999,0],[0,0.5]])
  F_dash = np.matrix([[1,1]])   
  for proc_noise_i in range(discretisation):
   proc_noise_std = float(proc_noise_i + 1) / (discretisation - 1)
   for obs_noise_i in range(discretisation):
    obs_noise_std  = float(obs_noise_i + 1) / (discretisation - 1)
  
    for runNo in range(noRuns):  
     sys = dynamical_system(G,np.zeros((2,1)),F_dash,np.zeros((1,1)),
      process_noise='gaussian',
      observation_noise='gaussian', 
      process_noise_std=proc_noise_std, 
      observation_noise_std=obs_noise_std,
      timevarying_multiplier_b = None)
     inputs = np.zeros(T)
     sys.solve([[1],[1]],inputs,T)
     Y = [i[0,0] for i in sys.outputs]
     #pdb.set_trace()
     ############################################
  
     ########## PRE-COMPUTE FILTER PARAMS ###################
     n = G.shape[0]
     m = F_dash.shape[0]

     W = proc_noise_std**2 * np.matrix(np.eye(n))
     V = obs_noise_std**2 * np.matrix(np.eye(m))
 
     #m_t = [np.matrix([[0],[0]])] 
     C = [np.matrix(np.eye(2))]
     R = []
     Q = []
     A = [] 
     Z = []


     for t in range(T):
        R.append(G * C[-1] * G.transpose() + W)
        Q.append(F_dash * R[-1] * F_dash.transpose() + V)
        A.append(R[-1]*F_dash.transpose()*np.linalg.inv(Q[-1]))
       
        C.append(R[-1] - A[-1]*Q[-1]*A[-1].transpose() )
        #Z.append(G*( np.eye(2) - F_dash.transpose()*A[-1].transpose() ))
        Z.append(G*( np.eye(2) - A[-1] * F_dash ))

     #PREDICTION
     Y_pred = []
     Y_kalman = []
     for t in range(T):
            Y_pred_term1 = F_dash * G * A[t] * sys.outputs[t]
            if t==0:
                Y_pred.append(Y_pred_term1)
                Y_kalman.append(Y_pred_term1)
                continue
            acc = 0
            for j in range(min(t,s)+1):
                for i in range(j+1):
                    if i==0: 
                        ZZ=Z[t-i]
                        continue
                    ZZ = ZZ*Z[t-i]
                acc += ZZ * G * A[t-j-1] * Y[t-j-1]
            Y_pred.append(Y_pred_term1 + F_dash*acc)
            accKalman = 0
            for j in range(t+1):
                for i in range(j+1):
                    if i==0: 
                        ZZ=Z[t-i]
                        continue
                    ZZ = ZZ*Z[t-i]
                accKalman += ZZ * G * A[t-j-1] * Y[t-j-1]
            Y_kalman.append(Y_pred_term1 + F_dash*accKalman)
     data[proc_noise_i][obs_noise_i] += np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(len(Y))])
     diffHere = np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(len(Y))]) 
     #print(Y_kalman[0][0,0])
     diffHere -= np.linalg.norm([Y_kalman[i][0,0] - Y[i] for i in range(min(len(Y),len(Y_kalman)))])
     #print(diffHere)
     diff[proc_noise_i][obs_noise_i] += diffHere
     #print(len(Y))
     #print(len(Y_kalman))
     errKalman[proc_noise_i][obs_noise_i] += pow(np.linalg.norm([Y_kalman[i][0,0] - Y[i] for i in range(min(len(Y),len(Y_kalman)))]), 2)
     errAR[proc_noise_i][obs_noise_i] += pow(np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(len(Y))]), 2)
  
  data = data / noRuns
  fig, ax = plt.subplots()
  tickLabels = [str(float(i+1) / 10) for i in range(11)]
  im, cbar = heatmap(data, tickLabels, tickLabels, ax=ax, cmap="YlGn", cbarlabel="Avg. RMSE of AR(%i), %s runs" % (s+1, noRuns))
  plt.ylabel('Variance of process noise')
  plt.xlabel('Variance of observation noise')
  fig.tight_layout()
  plt.savefig(pp, format='pdf')   
  #plt.show()
 
  diff = diff / noRuns
  fig, ax = plt.subplots()
  tickLabels = [str(float(i+1) / 10) for i in range(11)]
  im, cbar = heatmap(diff, tickLabels, tickLabels, ax=ax, cmap="YlOrRd", cbarlabel="Avg. diff. in RMSEs of AR(%i) and Kalman filter, %s runs" % (s+1, noRuns))
  plt.ylabel('Variance of process noise')
  plt.xlabel('Variance of observation noise')
  fig.tight_layout()
  plt.savefig(pp, format='pdf')   
  #plt.show()
  
  ratio = pow(errKalman / errAR, 2)
  fig, ax = plt.subplots()
  tickLabels = [str(float(i+1) / 10) for i in range(11)]
  im, cbar = heatmap(ratio, tickLabels, tickLabels, ax=ax, cmap="PuBu", cbarlabel="Ratios of agg. errors of Kalman and AR(%i), %s runs" % (s+1, noRuns))
  plt.ylabel('Variance of process noise')
  plt.xlabel('Variance of observation noise')
  fig.tight_layout()
  plt.savefig(pp, format='pdf')     
  
 pp.close()





def testImpactOfS(T = 200, noRuns = 100, sMax = 15):
 
 if sMax > T:
     print("The number of s to test must be less than the horizon T.") 
     exit()

 filename = 'impacts.pdf'
 pp = PdfPages(filename)
 
 for (proc_noise_std, obs_noise_std, linestyle) in [ (0.1, 0.1, "dotted"), (0.1, 1.0, "dashdot"),  (1.0, 0.1, "dashed"), (1.0, 1.0, "solid") ]:
  errAR = np.zeros((sMax+1, noRuns))
  ################# SYSTEM ###################        
  G = np.matrix([[0.999,0],[0,0.5]])
  F_dash = np.matrix([[1,1]])   
  for s in range(1, sMax):
  
    for runNo in range(noRuns):  
     sys = dynamical_system(G,np.zeros((2,1)),F_dash,np.zeros((1,1)),
      process_noise='gaussian',
      observation_noise='gaussian', 
      process_noise_std=proc_noise_std, 
      observation_noise_std=obs_noise_std,
      timevarying_multiplier_b = None)
     inputs = np.zeros(T)
     sys.solve([[1],[1]],inputs,T)
     Y = [i[0,0] for i in sys.outputs]
     #pdb.set_trace()
     ############################################
  
     ########## PRE-COMPUTE FILTER PARAMS ###################
     n = G.shape[0]
     m = F_dash.shape[0]

     W = proc_noise_std**2 * np.matrix(np.eye(n))
     V = obs_noise_std**2 * np.matrix(np.eye(m))
 
     #m_t = [np.matrix([[0],[0]])] 
     C = [np.matrix(np.eye(2))]
     R = []
     Q = []
     A = [] 
     Z = []


     for t in range(T):
        R.append(G * C[-1] * G.transpose() + W)
        Q.append(F_dash * R[-1] * F_dash.transpose() + V)
        A.append(R[-1]*F_dash.transpose()*np.linalg.inv(Q[-1]))
       
        C.append(R[-1] - A[-1]*Q[-1]*A[-1].transpose() )
        #Z.append(G*( np.eye(2) - F_dash.transpose()*A[-1].transpose() ))
        Z.append(G*( np.eye(2) - A[-1] * F_dash ))

     #PREDICTION
     Y_pred = []
     for t in range(T):
            Y_pred_term1 = F_dash * G * A[t] * sys.outputs[t]
            if t==0:
                Y_pred.append(Y_pred_term1)
                continue
            acc = 0
            for j in range(min(t,s)+1):
                for i in range(j+1):
                    if i==0: 
                        ZZ=Z[t-i]
                        continue
                    ZZ = ZZ*Z[t-i]
                acc += ZZ * G * A[t-j-1] * Y[t-j-1]
            Y_pred.append(Y_pred_term1 + F_dash*acc)
     errAR[s][runNo] = pow(np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(min(len(Y), len(Y_pred)))]), 2) / T


  error_AR1_mean = np.mean(errAR, 1)
  error_AR1_std = np.std(errAR, 1)
  print(len(error_AR1_mean))
  alphaValue = 0.2
  cAR1 = (proc_noise_std, obs_noise_std, 1.0/255)
  #plt.plot(range(1, sMax), error_AR1_mean[1:], label='AR(2)', color=cAR1, linewidth=2, antialiased = True)
  #plt.fill_between(range(1, sMax), (error_AR1_mean-error_AR1_std)[1:], (error_AR1_mean+error_AR1_std)[1:], alpha=alphaValue, edgecolor=cAR1, linewidth=2, antialiased=True) #transform=trans) #offset_position="data") alpha=alphaValue,   
  lab = "W = %.2f, V = %.2f" % (proc_noise_std, obs_noise_std)
  plt.plot(range(sMax+1)[1:-1], error_AR1_mean[1:-1], color=cAR1, linewidth=2, antialiased = True, label = lab, linestyle= linestyle)
  plt.fill_between(range(sMax+1)[1:-1], (error_AR1_mean-error_AR1_std)[1:-1], (error_AR1_mean+error_AR1_std)[1:-1], alpha=alphaValue, facecolor = cAR1, edgecolor=cAR1, linewidth=2, antialiased=True) #transform=trans) #offset_position="data") alpha=alphaValue,   
  plt.xlabel('Number s of auto-regressive terms, past the first one')
  plt.ylabel('Avg. error of AR(s), %i runs' % noRuns )
  plt.ylim(0, 1.5)
  plt.legend()
  plt.savefig(pp, format='pdf')     
  
 pp.close()


def testSeqD0():
    ts = time_series(matlabfile = './OARIMA_code_data/data/setting6.mat', varname="seq_d0")
    T = len(ts.outputs)
    testIdentification(ts, "seq0-complete", 1, T, 5, (2500.0, 5000.0), sequenceLabel = "seq_d0", haveSpectral = False)
    T = min(20000, len(ts.outputs))
    testIdentification(ts, "seq0-20000", 1, T, 5, (2500.0, 5000.0), sequenceLabel = "seq_d0", haveSpectral = False)    
    T = min(2000, len(ts.outputs))
    testIdentification(ts, "seq0-2000", 1, T, 5, (2500.0, 5000.0), 30, 45, sequenceLabel = "seq_d0", haveSpectral = False)
    T = min(200, len(ts.outputs))
    testIdentification(ts, "seq0-200", 1, T, 5, (2500.0, 5000.0), 30, 45, sequenceLabel = "seq_d0", haveSpectral = False)    
    T = min(100, len(ts.outputs))
    testIdentification(ts, "seq0-short-k5", 1, T, 5, (100.0, 1000.0, 2500.0, 10000.0), 27, 37, sequenceLabel = "seq_d0")
    #testIdentification(ts, "seq0-short-k50", 1, T, 50, 27, 37, sequenceLabel = "seq_d0")
    #testIdentification(ts, "seq0-short-k5", 1, T, 5, sequenceLabel = "seq_d0")
    #testIdentification(ts, "seq0-short-k50", 1, T, 50, sequenceLabel = "seq_d0")


def test_AR():
    ts = time_series(matlabfile = './OARIMA_code_data/data/setting6.mat', varname="seq_d0")
    T = min(100, len(ts.outputs))
    s=10
    D=10.
    theta = [0 for i in range(s)]

    for t in range(s,T):
        eta = pow(float(t),-0.5)

        Y = ts.outputs[t]

        loss = cost_AR(theta, Y, list(reversed(ts.outputs[t-s:t])))
        grad = gradient_AR(theta, Y, list(reversed(ts.outputs[t-s:t])))
       
        print("Loss: at time step %d :" % (t), loss)
        theta = [theta[i] -eta*grad[i] for i in range(len(theta))] #gradient step
        norm_theta = np.linalg.norm(theta)

        if norm_theta>D: theta = [D*i/norm_theta for i in theta] #projection step

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

def gradient_AR_test(component):

    theta = [1,2,3]
    args = [0 for i in range(2)]
    Y = 5
    past_obs = [4,5,6]

    h=1.
    db=10.
    J = cost_AR(theta, Y, past_obs)
    dJ = gradient_AR(theta, Y, past_obs)
    dJ0 = dJ[component]

    for i in range(10):
        theta_in = list(theta)
        theta_in[component] += h*db
        J_new = cost_AR(theta_in, Y, past_obs)
        print("h = %e; trunc = %.10e" % (h, J_new - J -dJ0*h*db ))
        h/=10


if __name__ == '__main__':
    try:
        close_all_figs()
        testIdentification2(T = 100, noRuns = 10, haveSpectral = True)
        testIdentification2(200, 10, haveSpectral = False)
        testIdentification2(500, noRuns = 100, sChoices = [1], haveKalman = True, haveSpectral = True)
        testNoiseImpact()
        #testImpactOfS()
        #timeSeqD0()
        #testSisoInvariantShort(100)
        #testIdentification2(100)
        #testSeqD0()
        #timeSeqD0()
        #testSeqD1()
        #testSeqD2()
        #testSisoInvariantLong()
        #testSYSID()
        #gradient_AR_test(0)
        #test_AR()
        #transition = np.matrix([[1.,-0.8],[-.6,.3]])
        #observation = np.matrix([[1.0,1.0]])
        #testIdentification2(20, noRuns = 100, sChoices = [1], haveKalman = True, haveSpectral = True, G = transition, F_dash = observation)
    except (KeyboardInterrupt, SystemExit):
            raise
    except:
        print(" Error: ")
        print(traceback.format_exc())
