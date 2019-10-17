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

    def logratio(self):
        """ Replaces the time series by a log-ratio of subsequent element therein."""
        T = len(self.outputs)
        newOutputs = []
        for (a, b) in zip(self.outputs[:T-1], self.outputs[1:T]):
            newOutputs.append( math.log(a / b) )
        self.outputs = newOutputs



