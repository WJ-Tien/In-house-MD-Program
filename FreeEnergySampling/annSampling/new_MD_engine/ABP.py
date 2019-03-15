#!/usr/bin/env python3
from mdlib.mdEngine import mdEngine
from mdlib.mdFileIO import mdFileIO
from mdlib.force import Force
from mdlib.customMathFunc import getIndices, paddingRighMostBins
from annlib.abpANN import trainingANN
import numpy as np
import tensorflow as tf
import time

class ABP(object):

  def __init__(self, input_mdp_file):

    self.IO              = mdFileIO()
    self.p               = self.IO.readParamFile(input_mdp_file) # p for md parameters
    self.bins            = np.linspace(-self.p["half_boxboundary"], self.p["half_boxboundary"], self.p["binNum"] + 1, dtype=np.float64)
    self.colvars_coord   = np.linspace(-self.p["half_boxboundary"], self.p["half_boxboundary"], self.p["binNum"] + 1, dtype=np.float64)

    self.mdInitializer   = mdEngine(self.p["nparticle"], self.p["box"], self.p["kb"],\
                                    self.p["time_step"], self.p["temperature"], self.p["ndims"],\
                                    self.p["mass"], self.p["thermoStatFlag"], self.getCurrentForce, self.p["frictCoeff"])

    self.initializeForce = Force(self.p["kb"], self.p["time_step"], self.p["temperature"], self.p["ndims"], self.p["mass"], self.p["thermoStatFlag"], self.p["frictCoeff"])

    # TODO initialize atom_coords in another module 
    self.current_coord   = np.zeros((self.p["nparticle"], self.p["ndims"]), dtype=np.float64)
    self.current_vel     = self.mdInitializer.genVelocity() 
    
    if self.p["ndims"] == 1:
      self.colvars_force    = np.zeros(len(self.bins), dtype=np.float64) 
      self.colvars_FreeE    = np.zeros(len(self.bins), dtype=np.float64) 
      self.colvars_FreeE_NN = np.zeros(len(self.bins), dtype=np.float64) 
      self.colvars_count    = np.zeros(len(self.bins), dtype=np.float64) 
      self.colvars_hist     = np.zeros(len(self.bins), dtype=np.float64) 
      self.biasingPotentialFromNN = np.zeros(len(self.bins), dtype=np.float64)

    if self.p["ndims"] == 2:
      self.colvars_force    = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_FreeE    = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_FreeE_NN = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_count    = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
      self.biasingPotentialFromNN = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64)

  def _histDistrRecord(self, coord_x, coord_y, d):

    if self.p["ndims"] == 1:
      self.colvars_count[getIndices(coord_x, self.bins)] += 1

    if self.p["ndims"] == 2:
      self.colvars_count[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] += 1

  def _forceDistrRecord(self, coord_x, updated_Fsys, coord_y, d):

    if self.p["ndims"] == 1:
      self.colvars_force[getIndices(coord_x, self.bins)] += updated_Fsys 

    if self.p["ndims"] == 2:
      self.colvars_force[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] += updated_Fsys 

  def _biasingPotential(self, coord_x, coord_y=None):
    if self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes":
      if self.p["ndims"] == 1:
        if self.p["init_frame"] <= self.p["trainingFreq"]: # initial sweep
          return 0 
        else:
          return self.biasingPotentialFromNN[getIndices(coord_x, self.bins)]

      if self.p["ndims"]== 2: 
        if self.p["init_frame"] <= self.p["traningFreq"]: # initial sweep
          return 0 
        else:
          return self.biasingPotentialFromNN[getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)]  
    else:
      return 0

  def _inverseGradient(self):
    """ cv == cartesian so return 1"""
    return 1

  def _Jacobian(self):
    """ cv == cartesian -> ln|J| = 0 so return 0"""
    return 0

  def _entropicCorrection(self):
    return self.p["kb"] * self.p["temperature"] * self._Jacobian()

  def _calBiasingForce(self, coord_x, coord_y, d):

    if self.p["ndims"] == 1:
      if self.colvars_count[getIndices(coord_x, self.bins)] == 0:
        return 0
      else:
        return -((self.colvars_force[getIndices(coord_x, self.bins)] / self.colvars_count[getIndices(coord_x, self.bins)] + self._entropicCorrection()) * self._inverseGradient())

    if self.p["ndims"] == 2:
      if self.colvars_count[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] == 0:
        return 0
      else:
        return -((self.colvars_force[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] / self.colvars_count[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] +\
                self._entropicCorrection()) * self._inverseGradient()) 

  def _abfDecorator(func):
    def _wrapper(self, coord_x, d, vel, coord_y):
      Fabf = func(self, coord_x, d, vel, coord_y)
      currentFsys = self.initializeForce.getForce(coord_x, d, vel, coord_y)
      self._forceDistrRecord(coord_x, currentFsys, coord_y, d)
      self._histDistrRecord(coord_x, coord_y, d)
      return Fabf + currentFsys # Fabf + currentFsys(unbiased)
    return _wrapper

  @_abfDecorator
  def getCurrentForce(self, coord_x, d, vel, coord_y): 

    if self.p["abfCheckFlag"] == "no" and self.p["nnCheckFlag"] == "no":
      Fabf = 0  

    elif self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes":

      if self.p["init_frame"] < self.p["trainingFreq"]:
        Fabf = self._calBiasingForce(coord_x, coord_y, d)

      else: # ANN takes over here

        if self.p["ndims"] == 1:
          Fabf = self.gradient[getIndices(coord_x, self.bins)]

        elif self.p["ndims"] == 2:
          Fabf = self.gradient[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)]
        
    return Fabf

  def _probability(self): 
    """ 1. unbias the historgram
        2. calculate the partition function
        3. calculate the probability and return it """

    rwHist = np.zeros(len(self.bins), dtype=np.float64) 
    maxValueOfBiasingPotential = np.amax(self.biasingPotentialFromNN)

    if self.p["ndims"] == 1:

      """
      for i in range(len(self.colvars_count)):
        rwHist[i] = (self.colvars_count[i]/2) * np.exp(self._biasingPotential(self.bins[i]) / self.p["kb"] / self.p["temperature"]) *\
                    np.exp(-maxValueOfBiasingPotential / self.p["kb"] / self.p["temperature"])
      """
      for i in range(len(self.colvars_hist)):
        rwHist[i] = self.colvars_hist[i] * np.exp(self._biasingPotential(self.bins[i]) / self.p["kb"] / self.p["temperature"]) *\
                    np.exp(-maxValueOfBiasingPotential / self.p["kb"] / self.p["temperature"])
      partitionFunc = np.sum(rwHist)
      probabilityDistr = rwHist / partitionFunc

    if self.p["ndims"] == 2: #TODO 2D
      pass

    return probabilityDistr 

  def getCurrentFreeEnergy(self):  
    self.colvars_FreeE = -self.p["kb"] * self.p["temperature"] * np.log(self._probability())
    self.colvars_FreeE[np.isneginf(self.colvars_FreeE)] = 0.0  # deal with log(0) = -inf
    self.colvars_FreeE[np.isinf(self.colvars_FreeE)] = 0.0  # deal with inf
    self.colvars_FreeE = paddingRighMostBins(self.p["ndims"], self.colvars_FreeE) # for the sake of PBC

  def _updateBiasingPotential(self):
    self.biasingPotentialFromNN = -self.colvars_FreeE_NN.copy() # phi(x) = -Fhat(x) 

  def _learningProxy(self):
    if self.p["nnCheckFlag"] == "yes":
      if self.p["init_frame"] % self.p["trainingFreq"] == 0 and self.p["init_frame"] != 0: 
        output = trainingANN("loss.dat", "hyperparam.dat", self.p["ndims"], len(self.bins)) 

        if self.p["init_frame"] < self.p["trainingFreq"] * self.p["switchSteps"]:
          self.colvars_FreeE_NN, self.gradient = \
          output.training(self.colvars_coord, self.colvars_FreeE, self.p["earlyLearningRate"], self.p["earlyRegularCoeff"], self.p["earlyEpoch"], self.p["nnOutputFreq"]) 

        else:
          self.colvars_FreeE_NN, self.gradient = \
          output.training(self.colvars_coord, self.colvars_FreeE, self.p["lateLearningRate"], self.p["lateRegularCoeff"], self.p["lateEpoch"], self.p["nnOutputFreq"]) 
      #TODO instant output

  def mdrun(self):

    init_real_world_time = time.time()

    # pre-processing
    lammpstrj      = open("m%.1f_T%.5f_gamma%.1f_len_%d.lammpstrj" %(self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"]), "w")
    forceOnCVs     = open("Force_m%.1fT%.5f_gamma%.1f_len_%d.dat" %(self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"]), "w")
    freeEOnCVs     = open("FreeE_m%.1fT%.5f_gamma%.1f_len_%d.dat" %(self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"]), "w")
    histogramOnCVs = open("Hist_m%.1fT%.5f_gamma%.1f_len_%d.dat" %(self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"]), "w")

    # Start of the simulation
    # the first frame
    self.IO.writeParams(self.p)
    self.IO.lammpsFormatColvarsOutput(self.p["ndims"], self.p["nparticle"], self.p["half_boxboundary"], self.p["init_frame"], self.current_coord, lammpstrj, self.p["writeFreq"]) 
    self.IO.printCurrentStatus(self.p["init_frame"], init_real_world_time)  
    
    # the rest of the frames
    while self.p["init_frame"] < self.p["total_frame"]: 

      self.p["init_frame"] += 1
      self.IO.printCurrentStatus(self.p["init_frame"], init_real_world_time)  

      self.mdInitializer.checkTargetTemperature(self.current_vel, self.p["init_frame"], self.p["total_frame"])

      if self.p["init_frame"] % self.p["trainingFreq"] == 0 and self.p["init_frame"] != 0 and self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes":
        self.getCurrentFreeEnergy()
        self._learningProxy()
        self._updateBiasingPotential()

      elif self.p["init_frame"] == self.p["total_frame"] and self.p["abfCheckFlag"] == "no" and self.p["abfCheckFlag"] == "no":
        self.getCurrentFreeEnergy()
      
      self.mdInitializer.velocityVerletSimple(self.current_coord, self.current_vel) 

      for n in range(self.p["nparticle"]):
        if self.p["ndims"] == 1:
          self.colvars_hist[getIndices(self.current_coord[n][0], self.bins)] += 1
        elif self.p["ndims"] == 2:
          pass

      self.colvars_hist = paddingRighMostBins(self.p["ndims"], self.colvars_hist)

      self.IO.lammpsFormatColvarsOutput(self.p["ndims"], self.p["nparticle"], self.p["half_boxboundary"], self.p["init_frame"], self.current_coord, lammpstrj, self.p["writeFreq"]) 
    # End of simulation

    # post-processing
    #probability = self.colvars_count / (np.sum(self.colvars_count) / self.p["ndims"])   # both numerator and denominator should actually divided by two but these would be cacncelled
    #probability = paddingRighMostBins(self.p["ndims"], probability) 
    probability = (self.colvars_hist / np.sum(self.colvars_hist))
    self.IO.propertyOnColvarsOutput(self.p["ndims"], self.bins, probability, self.colvars_count, histogramOnCVs)

    self.colvars_force = (self.colvars_force / self.colvars_count)
    self.colvars_force[np.isnan(self.colvars_force)] = 0
    self.colvars_force = paddingRighMostBins(self.p["ndims"], self.colvars_force) 
    self.IO.propertyOnColvarsOutput(self.p["ndims"], self.bins, self.colvars_force, self.colvars_count, forceOnCVs)

    if self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes":  
      self.IO.propertyOnColvarsOutput(self.p["ndims"], self.bins, self.colvars_FreeE_NN, self.colvars_count, freeEOnCVs)

    else:
      self.IO.propertyOnColvarsOutput(self.p["ndims"], self.bins, self.colvars_FreeE, self.colvars_count, freeEOnCVs)
        
    self.IO.closeAllFiles(lammpstrj, forceOnCVs, freeEOnCVs, histogramOnCVs)

if __name__ == "__main__":
  ABP("in.mdp").mdrun()
