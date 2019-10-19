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

# IBM-Review-Requirement: Art30.3
# Please note that the following code was developed for the project VaVeL at IBM Research 
# -- Ireland, funded by the European Union under the Horizon 2020 Program. 
# The project started on December 1st, 2015 and was completed by December 1st,
# 2018. Thus, in accordance with Article 30.3 of the Multi-Beneficiary General 
# Model Grant Agreement of the Program, the above limitations are in force.
# For further details please contact Jakub Marecek (jakub.marecek@ie.ibm.com), 
# or Gal Weiss (wgal@ie.ibm.com).

# If you use this code, please cite our paper:
# @inproceedings{kozdoba2018,
#  title={On-Line Learning of Linear Dynamical Systems: Exponential Forgetting in Kalman Filters},
#  author={Kozdoba, Mark and Marecek, Jakub and Tchrakian, Tigran and Mannor, Shie},
#  booktitle = {The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19)},
#  note={arXiv preprint arXiv:1809.05870},
#  year={2019}
#}

from __future__ import print_function

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
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

verbose = False

from onlinelds import *
from inputlds import *

def close_all_figs():
    plt.close('all')

def testIdentification(sys, filenameStub = "test", noRuns = 2, T = 100, k = 5, etaZeros = None, ymin = None, ymax = None, sequenceLabel = None, haveSpectral = True):
    """ noRuns is the number of runs, T is the time horizon, k is the number of filters, """
    
    if k>T:
        print("Number of filters (k) must be less than or equal to the number of time-steps (T).")
        exit()
    if not etaZeros:
        etaZeros = [1.0, 2500.0]
    print("etaZeros:")
    print(etaZeros)

    filename = './outputs/' + filenameStub+'.pdf'
    pp = PdfPages(filename)
    
    error_AR_data = None
    error_spec_data = None
    error_persist_data = None

    for i in range(noRuns):
      print("run %i" % i)
      inputs = np.zeros(T)
      sys.solve([[1],[0]],inputs,T)
      
      if haveSpectral:
        predicted_spectral, M, error_spec, error_persist = wave_filtering_SISO_ftl(sys, T, k)
        if error_spec_data is None: error_spec_data = error_spec
        else: error_spec_data = np.vstack((error_spec_data, error_spec))
        if error_persist_data is None: error_persist_data = error_persist
        else: error_persist_data = np.vstack((error_persist_data, error_persist))        
      
      for etaZero in etaZeros:  
        error_AR = np.zeros(T)
        predicted_AR = np.zeros(T)
        s=2
        D=1.
        theta = [0 for i in range(s)]
        for t in range(s,T):
            eta = pow(float(t),-0.5) / etaZero
            Y = sys.outputs[t]
            loss = cost_AR(theta, Y, list(reversed(sys.outputs[t-s:t])))
            error_AR[t] = pow(loss, 0.5)
            grad = gradient_AR(theta, Y, list(reversed(sys.outputs[t-s:t])))       
            #print("Loss: at time step %d :" % (t), loss)
            theta = [theta[i] -eta*grad[i] for i in range(len(theta))] #gradient step
            norm_theta = np.linalg.norm(theta)
            if norm_theta>D: theta = [D*i/norm_theta for i in theta] #projection step
            predicted_AR[t] = np.dot(list(reversed(sys.outputs[t-s:t])),theta)
            
        if error_AR_data is None: error_AR_data = error_AR
        else: error_AR_data = np.vstack((error_AR_data, error_AR))        
    
        p1 = plt.figure()
        if ymax and ymin: plt.ylim(ymin, ymax)
        if sum(inputs[1:]) > 0: plt.plot(inputs[1:], label='Input')
        if sequenceLabel: plt.plot([float(i) for i in sys.outputs][1:], label=sequenceLabel, color='#000000', linewidth=2, antialiased = True)
        else: plt.plot([float(i) for i in sys.outputs][1:], label='Output', color='#000000', linewidth=2, antialiased = True)
        #plt.plot([-i for i in predicted_output], label='Predicted output') #for some reason, usual way produces -ve estimate
        if haveSpectral: 
            plt.plot([i for i in predicted_spectral], label='Spectral')
        #lab = 'AR(3) / OGD, c_0 = ' + str(etaZero)
        lab = "AR(" + str(s) + "), c = " + str(int(etaZero))
        plt.plot(predicted_AR, label = lab)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Output')         
        p1.show()
        p1.savefig(pp, format='pdf')

        p2 = plt.figure()
        plt.ylim(0, 20)
        if haveSpectral: 
            plt.plot(error_spec, label='Spectral')
            plt.plot(error_persist, label='Persistence')
        plt.plot(error_AR, label=lab)
        plt.legend()
        p2.show()
        plt.xlabel('Time')
        plt.ylabel('Error')           
        p2.savefig(pp, format='pdf')

    error_AR_mean = np.mean(error_AR_data, 0)
    error_AR_std = np.std(error_AR_data, 0)
    if haveSpectral:
      error_spec_mean = np.mean(error_spec_data, 0)
      error_spec_std = np.std(error_spec_data, 0)
      error_persist_mean = np.mean(error_persist_data, 0)
      error_persist_std = np.std(error_persist_data, 0)    

    p3 = plt.figure()
    if ymax and ymin: plt.ylim(ymin, ymax)    
    if haveSpectral:
      plt.plot(error_spec_mean, label='Spectral', color='#1B2ACC', linewidth=2, antialiased = True)
      plt.fill_between(range(0,T-1), error_spec_mean-error_spec_std, error_spec_mean+error_spec_std, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
      linewidth=1, antialiased=True)
      plt.plot(error_persist_mean, label='Persistence', color='#CC1B2A', linewidth=2, antialiased = True)
      plt.fill_between(range(0,T-1), error_persist_mean-error_persist_std, error_persist_mean+error_persist_std, alpha=0.2, edgecolor='#CC1B2A', facecolor='#FF0800',
      linewidth=1, antialiased=True)

    cAR1 = (42.0/255, 204.0 / 255.0, 1.0/255)
    bAR1 = (1.0, 204.0 / 255.0, 0.0) # , alphaValue
    plt.ylim(0, 20)
    plt.plot(error_AR_mean, label='AR(3)', color=cAR1, linewidth=2, antialiased = True)
    plt.fill_between(range(0,T), error_AR_mean-error_AR_std, error_AR_mean+error_AR_std, alpha=0.2, edgecolor=cAR1, facecolor=bAR1,
    linewidth=1, antialiased=True)    
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Error')    
    p3.savefig(pp, format='pdf')

    pp.close()
    print("See the output in " + filename)

def testIdentification2(T = 100, noRuns = 10, sChoices = [15,3,1], haveKalman = False, haveSpectral = True, G = np.matrix([[0.999,0],[0,0.5]]), F_dash = np.matrix([[1,1]]), sequenceLabel = ""):
  if haveKalman: sChoices = sChoices + [T]
  if len(sequenceLabel) > 0: sequenceLabel = " (" + sequenceLabel + ")"

  if noRuns < 2:
    print("Number of runs has to be larger than 1.")
    exit()

  filename = './outputs/AR.pdf'
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
            plt.plot([i[0,0] for i in Y_pred], label="Kalman" + sequenceLabel, color=(42.0/255.0, 204.0 / 255.0, 200.0/255.0), linewidth=2, antialiased = True)
        else:
            plt.plot([i[0,0] for i in Y_pred], label='AR(%i)' % (s+1)  + sequenceLabel, color=(42.0/255.0, 204.0 / 255.0, float(min(255.0,s))/255.0), linewidth=2, antialiased = True)
        
        plt.xlabel('Time')
        plt.ylabel('Prediction') 
        
    
    if haveSpectral:    
      predicted_output, M, error_spec, error_persist = wave_filtering_SISO_ftl(sys, T, 5)
      plt.plot(predicted_output, label='Spectral' + sequenceLabel, color='#1B2ACC', linewidth=2, antialiased = True)
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
      plt.plot(range(0,Tlim), error_spec[:Tlim], label='Spectral' + sequenceLabel, color='#1B2ACC', linewidth=2, antialiased = True)
      plt.fill_between(range(0,Tlim), (error_spec_mean-error_spec_std)[:Tlim], (error_spec_mean+error_spec_std)[:Tlim], alpha=alphaValue, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=1, antialiased=True)
      plt.plot(range(0,Tlim), error_persist[:Tlim], label='Persistence' + sequenceLabel, color='#CC1B2A', linewidth=2, antialiased = True)
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
    plt.plot(error_AR1_mean[:Tlim], label='AR(2)' + sequenceLabel, color=cAR1, linewidth=2, antialiased = True)
    plt.fill_between(range(0,Tlim), (error_AR1_mean-error_AR1_std)[:Tlim], (error_AR1_mean+error_AR1_std)[:Tlim], alpha=alphaValue, edgecolor=cAR1, facecolor=bAR1, linewidth=1, antialiased=True) #transform=trans) #offset_position="data") alpha=alphaValue, 
    if haveKalman:
      cK = (42.0/255.0, 204.0 / 255.0, 200.0/255.0)
      bK = (1.0, 204.0 / 255.0, 200.0/255.0) # alphaValue
      print(cK)
      print(bK)
      plt.plot(error_Kalman_mean[:Tlim], label='Kalman' + sequenceLabel, color=cK, linewidth=2, antialiased = True)
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
 filename = './outputs/noise.pdf'
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

 filename = './outputs/impacts.pdf'
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


def testSeqD0(noRuns = 100):
  plain = False
  lr = True
    
  if plain: 
    ts = time_series(matlabfile = './OARIMA_code_data/data/setting6.mat', varname="seq_d0")
    T = len(ts.outputs)
    testIdentification(ts, "seq0-complete", noRuns, T, 5, sequenceLabel = "seq_d0", haveSpectral = False)
    T = min(20000, len(ts.outputs))
    testIdentification(ts, "seq0-20000", noRuns, T, 5, sequenceLabel = "seq_d0", haveSpectral = False)    
    T = min(2000, len(ts.outputs))
    testIdentification(ts, "seq0-2000", noRuns, T, 5, sequenceLabel = "seq_d0", haveSpectral = False)
    T = min(200, len(ts.outputs))
    testIdentification(ts, "seq0-200", noRuns, T, 5, sequenceLabel = "seq_d0", haveSpectral = False)    
    T = min(100, len(ts.outputs))
    testIdentification(ts, "seq0-short-k5", 1, T, 5, sequenceLabel = "seq_d0")
    #testIdentification(ts, "seq0-short-k50", 1, T, 50, 27, 37, sequenceLabel = "seq_d0")
    #testIdentification(ts, "seq0-short-k5", 1, T, 5, sequenceLabel = "seq_d0")
    #testIdentification(ts, "seq0-short-k50", 1, T, 50, sequenceLabel = "seq_d0")
  if lr:
    ts = time_series(matlabfile = './OARIMA_code_data/data/setting6.mat', varname="seq_d0")
    ts.logratio()
    T = len(ts.outputs) # has to go after the log-ratio truncation by one
    testIdentification(ts, "logratio-complete", noRuns, T, 5, sequenceLabel = "lr_d0", haveSpectral = False)
    T = min(20000, len(ts.outputs))
    testIdentification(ts, "logratio-20000", noRuns, T, 5,  sequenceLabel = "lr_d0", haveSpectral = False)    
    T = min(2000, len(ts.outputs))
    testIdentification(ts, "logratio-2000", noRuns, T, 5, sequenceLabel = "lr_d0", haveSpectral = False)
    T = min(200, len(ts.outputs))
    testIdentification(ts, "logratio-200", noRuns, T, 5, sequenceLabel = "lr_d0", haveSpectral = False)    
    T = min(100, len(ts.outputs))
    testIdentification(ts, "logratio-short-k5", noRuns, T, 5, sequenceLabel = "lr_d0")

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

version = "FinalAAAI"
version = "Working"
version = "Extended"

if __name__ == '__main__':
    try:
        close_all_figs()
        if version == "Extended":
          # The following calls adds the plots for the extended version
          testSeqD0()
        if version == "FinalAAAI":
          # These calls produce the AAAI 2019 figures (8-page version)
          testIdentification2(500, noRuns = 100, sChoices = [1], haveKalman = True, haveSpectral = True)
          testNoiseImpact()
          testImpactOfS()
        if version == "Working":
            # These calls produce illuminating plots, which did not make it into the final 8-page version of the paper.  
            None
            #testIdentification2(T = 100, noRuns = 10, haveSpectral = True)
            #testIdentification2(200, 10, haveSpectral = False)
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

