#!/usr/bin/env python3
from mdlib.mdEngine import mdEngine
from mdlib.mdFileIO import mdFileIO
from mdlib.force import Force
from mdlib.render import rendering
from mdlib.customMathFunc import getIndices, paddingRighMostBins
from annlib.abpANN import trainingANN
import numpy as np
import tensorflow as tf
import time
import copy 

class ABP(object):

  def __init__(self, input_mdp_file):

    self.IO              = mdFileIO()
    self.p               = self.IO.readParamFile(input_mdp_file) # p for md parameters
    self.binw            = 2 * self.p["half_boxboundary"] / self.p["binNum"]
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
      self.biasingPotentialFrom = np.zeros(len(self.bins), dtype=np.float64)
      self.biasingPotentialFromNN = np.zeros(len(self.bins), dtype=np.float64)

    if self.p["ndims"] == 2:
      self.colvars_force    = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_FreeE    = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_FreeE_NN = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_count    = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_hist     = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64) 
      self.biasingPotentialFrom = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64)
      self.biasingPotentialFromNN = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64)

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
        if self.p["init_frame"] <= self.p["trainingFreq"]: # initial sweep
          return 0 
        else:
          return self.biasingPotentialFromNN[getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)]  

    elif self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "no":

        if self.p["init_frame"] <= self.p["trainingFreq"]: # initial sweep
          return 0 
        else:
          return self.biasingPotentialFrom[getIndices(coord_x, self.bins)]

        if self.p["ndims"]== 2: 
          if self.p["init_frame"] <= self.p["trainingFreq"]: # initial sweep
            return 0 
          else:
            return self.biasingPotentialFrom[getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)]  
    else:
      return 0

  def _calBiasingForce(self, coord_x, coord_y, d):

    if self.p["abfCheckFlag"] == "yes":
      bsForce = copy.deepcopy(self.colvars_FreeE)

      if self.p["ndims"] == 1:
        bsForce = np.diff(bsForce)
        bsForce = np.append(bsForce, bsForce[-1]) # padding to the right legnth
        bsForce = (bsForce / self.binw)
        return bsForce[getIndices(coord_x, self.bins)]

      if self.p["ndims"] == 2:

        if d == 0:
          bsForce = np.diff(bsForce, axis=0) #axis=0 for x; axis=1 for y
          bsForce = np.append(bsForce, [bsForce[-1, :]], axis=0) # padding to the right length
          bsForce = (bsForce / self.binw)

        elif d == 1:
          bsForce = np.diff(bsForce, axis=1) #axis=0 for x; axis=1 for y
          bsForce = np.append(bsForce, bsForce[:, -1][:, np.newaxis], axis=1) # padding to the right length
          bsForce = (bsForce / self.binw)
        return bsForce[getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)]

    else:
      return 0
             
  def _abpDecorator(func):
    def _wrapper(self, coord_x, d, vel, coord_y):
      Fabf = func(self, coord_x, d, vel, coord_y)
      currentFsys = self.initializeForce.getForce(coord_x, d, vel, coord_y)
      self._forceDistrRecord(coord_x, currentFsys, coord_y, d)
      self._histDistrRecord(coord_x, coord_y, d)
      return Fabf + currentFsys # Fabf + currentFsys(unbiased)
    return _wrapper

  @_abpDecorator
  def getCurrentForce(self, coord_x, d, vel, coord_y): 

    if self.p["abfCheckFlag"] == "no" and self.p["nnCheckFlag"] == "no":
      Fabf = 0  

    elif self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "no":
      Fabf = self._calBiasingForce(coord_x, coord_y, d)

    elif self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes":

      if self.p["init_frame"] <= self.p["trainingFreq"]:
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

    if self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes":
      maxValueOfBiasingPotential = np.amax(self.biasingPotentialFromNN)

    elif self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "no":
      maxValueOfBiasingPotential = np.amax(self.biasingPotentialFrom)

    else:
      maxValueOfBiasingPotential = 0

  
    if self.p["ndims"] == 1:
      rwHist = np.zeros(len(self.bins), dtype=np.float64) 

      """
      for i in range(len(self.colvars_count)):
        rwHist[i] = (self.colvars_count[i]/2) * np.exp(self._biasingPotential(self.bins[i]) / self.p["kb"] / self.p["temperature"]) *\
                    np.exp(-maxValueOfBiasingPotential / self.p["kb"] / self.p["temperature"])
      """
      for i in range(len(self.colvars_hist)):
        rwHist[i] = self.colvars_hist[i] * np.exp(self._biasingPotential(self.bins[i]) / self.p["kb"] / self.p["temperature"]) *\
                    np.exp(-maxValueOfBiasingPotential / self.p["kb"] / self.p["temperature"])

    if self.p["ndims"] == 2:
      rwHist = np.zeros((len(self.bins),len(self.bins)), dtype=np.float64) 
      for i in range(len(self.colvars_hist)):
        for j in range(len(self.colvars_hist)):
          rwHist[i][j] = self.colvars_hist[i][j] * np.exp(self._biasingPotential(self.bins[i], self.bins[j]) / self.p["kb"] / self.p["temperature"]) *\
                                                   np.exp(-maxValueOfBiasingPotential / self.p["kb"] / self.p["temperature"])
    partitionFunc = np.sum(rwHist)
    probabilityDistr = rwHist / partitionFunc

    return probabilityDistr 

  def getCurrentFreeEnergy(self):  
    self.colvars_FreeE = -self.p["kb"] * self.p["temperature"] * np.log(self._probability())
    self.colvars_FreeE[np.isneginf(self.colvars_FreeE)] = 0.0  # deal with log(0) = -inf
    self.colvars_FreeE[np.isinf(self.colvars_FreeE)] = 0.0  # deal with inf
    self.colvars_FreeE = paddingRighMostBins(self.colvars_FreeE)

  def _updateBiasingPotential(self):
    self.biasingPotentialFromNN = -copy.deepcopy(self.colvars_FreeE_NN) # phi(x) = -Fhat(x)  for ANN
    self.biasingPotentialFrom = -copy.deepcopy(self.colvars_FreeE) # phi(x) = -Fhat(x) for non-ANN

  def _learningProxy(self):
    if self.p["nnCheckFlag"] == "yes":
      if self.p["init_frame"] % self.p["trainingFreq"] == 0 and self.p["init_frame"] != 0: 
        output = trainingANN("loss.dat", "hyperparam.dat", self.p["ndims"], len(self.bins), self.binw) 

        if self.p["init_frame"] < self.p["trainingFreq"] * self.p["switchSteps"]:
          self.colvars_FreeE_NN, self.gradient = \
          output.training(self.colvars_coord, self.colvars_FreeE, self.p["earlyLearningRate"], self.p["earlyRegularCoeff"], self.p["earlyEpoch"], self.p["nnOutputFreq"]) 

        else:
          self.colvars_FreeE_NN, self.gradient = \
          output.training(self.colvars_coord, self.colvars_FreeE, self.p["lateLearningRate"], self.p["lateRegularCoeff"], self.p["lateEpoch"], self.p["nnOutputFreq"]) 

  def _resetColvarsHist(self):
    self.colvars_hist.fill(0) 

  def mdrun(self):

    init_real_world_time = time.time()

    # pre-processing
    lammpstrj      = open("m%.1f_T%.5f_gamma%.1f_len_%d.lammpstrj" %(self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"]), "w")
    forceOnCVs     = open("Force_m%.1fT%.5f_gamma%.1f_len_%d.dat" %(self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"]), "w")
    freeEOnCVs     = open("FreeE_m%.1fT%.5f_gamma%.1f_len_%d.dat" %(self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"]), "w")
    histogramOnCVs = open("Hist_m%.1fT%.5f_gamma%.1f_len_%d.dat" %(self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"]), "w")
    annABP         = open("instantFreeEWANN_" + str(self.p["ndims"]) + "D.dat", "a")
    convABP        = open("instantFreeEWOANN_" + str(self.p["ndims"]) + "D.dat", "a")
    tempHist       = open("tempHist.dat", "a")
    tempBsp        = open("tempBsp.dat", "a")

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

      if self.p["init_frame"] % self.p["trainingFreq"] == 0 and self.p["init_frame"] != 0 and self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes": # for ANN-ABP
        self.getCurrentFreeEnergy()
        self._learningProxy()
        self._updateBiasingPotential()
        self.IO.certainFrequencyOutput(self.colvars_coord, self.colvars_FreeE_NN, self.colvars_hist, self.p["init_frame"], self.p["certainOutFreq"], annABP)
        self.IO.certainFrequencyOutput(self.colvars_coord, self.colvars_FreeE, self.colvars_hist, self.p["init_frame"], self.p["certainOutFreq"], convABP)
        self.IO.certainFrequencyOutput(self.colvars_coord, self.colvars_hist/np.sum(self.colvars_hist), self.colvars_hist, self.p["init_frame"], self.p["certainOutFreq"], tempHist)
        self.IO.certainFrequencyOutput(self.colvars_coord, self.biasingPotentialFromNN, self.colvars_hist, self.p["init_frame"], self.p["certainOutFreq"], tempBsp)
        if self.p["init_frame"] == self.p["total_frame"]: # the last frame
          pass 
        else:
          self._resetColvarsHist() # zero the histogram
      

      if self.p["init_frame"] % self.p["trainingFreq"] == 0 and self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "no" and self.p["init_frame"] != 0: #for regular ABP
        self.getCurrentFreeEnergy()
        self._updateBiasingPotential()
        self.IO.certainFrequencyOutput(self.colvars_coord, self.colvars_FreeE, self.colvars_hist, self.p["init_frame"], self.p["certainOutFreq"], convABP)
        self.IO.certainFrequencyOutput(self.colvars_coord, self.colvars_hist, self.colvars_hist, self.p["init_frame"], self.p["certainOutFreq"], tempHist)
        self.IO.certainFrequencyOutput(self.colvars_coord, self.biasingPotentialFrom, self.colvars_hist, self.p["init_frame"], self.p["certainOutFreq"], tempBsp)
        if self.p["init_frame"] == self.p["total_frame"]: # the last frame
          pass 
        else:
          self._resetColvarsHist() # zero the histogram

      if self.p["init_frame"] == self.p["total_frame"]: # the last frame
        self.getCurrentFreeEnergy()
      

      self.mdInitializer.velocityVerletSimple(self.current_coord, self.current_vel) 

      for n in range(self.p["nparticle"]): #TODO
        if self.p["ndims"] == 1:
          self.colvars_hist[getIndices(self.current_coord[n][0], self.bins)] += 1
        elif self.p["ndims"] == 2:
          self.colvars_hist[getIndices(self.current_coord[n][0], self.bins)][getIndices(self.current_coord[n][1], self.bins)] += 1

      self.IO.lammpsFormatColvarsOutput(self.p["ndims"], self.p["nparticle"], self.p["half_boxboundary"], self.p["init_frame"], self.current_coord, lammpstrj, self.p["writeFreq"]) 
    # End of simulation

    # post-processing
    #probability = self.colvars_count / (np.sum(self.colvars_count) / self.p["ndims"])   # both numerator and denominator should actually divided by two but these would be cacncelled
    #probability = paddingRighMostBins(self.p["ndims"], probability) 
    probability = copy.deepcopy((self.colvars_hist / np.sum(self.colvars_hist)))
    probability = paddingRighMostBins(probability)
    self.IO.propertyOnColvarsOutput(self.bins, probability, self.colvars_hist, histogramOnCVs)

    self.colvars_force = (self.colvars_force / self.colvars_count)
    self.colvars_force[np.isnan(self.colvars_force)] = 0
    self.colvars_force = paddingRighMostBins(self.colvars_force) 
    self.IO.propertyOnColvarsOutput(self.bins, self.colvars_force, self.colvars_count, forceOnCVs)

    if self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes":  
      self.IO.propertyOnColvarsOutput(self.bins, self.colvars_FreeE_NN, self.colvars_hist, freeEOnCVs)

    else:
      self.IO.propertyOnColvarsOutput(self.bins, self.colvars_FreeE, self.colvars_hist, freeEOnCVs)
        
    if self.p["ndims"] == 2: 
      s = rendering(self.p["ndims"], self.p["half_boxboundary"], self.p["binNum"], self.p["temperature"])
      s.render(self.colvars_FreeE, name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "Usurface_Original" +str(self.p["ndims"])+"D"))
      s.render(self.colvars_force[0], name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "forceX_UnderABP" +str(self.p["ndims"])+"D"))
      s.render(self.colvars_force[1], name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "forceY_UnderABP" +str(self.p["ndims"])+"D"))
      if self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes":  
        s.render(self.colvars_FreeE_NN, name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "Usurface_NN" + str(self.p["ndims"])+"D"))

    # close files, mdkir and mv files
    self.IO.closeAllFiles(lammpstrj, forceOnCVs, freeEOnCVs, histogramOnCVs, annABP, convABP, tempHist, tempBsp)
    self.IO.makeDirAndMoveFiles(self.p["ndims"], self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"],\
                                self.p["abfCheckFlag"], self.p["nnCheckFlag"], __class__.__name__)




if __name__ == "__main__":
  ABP("in.ABP").mdrun()
