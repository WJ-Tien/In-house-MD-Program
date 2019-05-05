#!/usr/bin/env python3
from mdlib.mdEngine import mdEngine
from mdlib.mdFileIO import mdFileIO 
from mdlib.force import Force
from mdlib.customMathFunc import getIndices, paddingRightMostBin
from mdlib.render import rendering
from mdlib.integrator import integrator
from annlib.abfANN import trainingANN
import numpy as np
import tensorflow as tf
import time
import copy

class ABF(object):

  def __init__(self, input_mdp_file):

    self.IO              = mdFileIO() 
    self.p               = self.IO.readParamFile(input_mdp_file) # p for md parameters
    self.bins            = np.linspace(-self.p["half_boxboundary"], self.p["half_boxboundary"], self.p["binNum"] + 1, dtype=np.float64)
    self.colvars_coord   = np.linspace(-self.p["half_boxboundary"], self.p["half_boxboundary"], self.p["binNum"] + 1, dtype=np.float64)

    self.mdInitializer   = mdEngine(self.p["nparticle"], self.p["box"], self.p["kb"],\
                                    self.p["time_step"], self.p["temperature"], self.p["ndims"],\
                                    self.p["mass"], self.p["thermoStatFlag"], self.getCurrentForce, self.p["frictCoeff"])

    self.initializeForce = Force(self.p["kb"], self.p["time_step"], self.p["temperature"], self.p["ndims"], self.p["mass"], self.p["thermoStatFlag"], self.p["frictCoeff"])

    # init coord and vel 
    self.current_coord   = np.zeros((self.p["nparticle"], self.p["ndims"]), dtype=np.float64)
    self.current_vel     = self.mdInitializer.genVelocity() 
    # init coord and vel

    self.criteriaCounter = 0 
    self.criteriaFEBool  = 0
    
    if self.p["ndims"] == 1:
      self.colvars_force      = np.zeros(len(self.bins), dtype=np.float64) 
      self.colvars_force_tmp  = np.zeros(len(self.bins), dtype=np.float64) 
      self.colvars_force_NN   = np.zeros(len(self.bins), dtype=np.float64) 
      self.colvars_count      = np.zeros(len(self.bins), dtype=np.float64) 
      self.colvars_hist       = np.zeros(len(self.bins), dtype=np.float64) 
      self.criteria_hist      = np.zeros(len(self.bins), dtype=np.float64) 
      self.criteria_prev      = np.zeros(len(self.bins), dtype=np.float64) 
      self.criteria_curr      = np.zeros(len(self.bins), dtype=np.float64) 
      self.criteria_FreeE     = np.zeros(len(self.bins), dtype=np.float64)
      self.colvars_FreeE      = np.zeros(len(self.bins), dtype=np.float64) 
      self.colvars_FreeE_prev = np.zeros(len(self.bins), dtype=np.float64) 
      self.colvars_FreeE_curr = np.zeros(len(self.bins), dtype=np.float64) 

    if self.p["ndims"] == 2:
      self.colvars_force      = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_force_tmp  = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_force_NN   = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_count      = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_hist       = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64) 
      self.criteria_hist      = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64) 
      self.criteria_prev      = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64) 
      self.criteria_curr      = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64) 
      self.criteria_FreeE     = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64)
      self.colvars_FreeE      = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_FreeE_prev = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64) 
      self.colvars_FreeE_curr = np.zeros((len(self.bins), len(self.bins)), dtype=np.float64) 

  def _forceHistDistrRecord(self, coord_x, coord_y, d):

    if self.p["ndims"] == 1:
      self.colvars_count[getIndices(coord_x, self.bins)] += 1

    if self.p["ndims"] == 2:
      self.colvars_count[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] += 1

  def _forceDistrRecord(self, coord_x, updated_Fsys, coord_y, d):

    if self.p["ndims"] == 1:
      self.colvars_force[getIndices(coord_x, self.bins)] += updated_Fsys 

    if self.p["ndims"] == 2:
      self.colvars_force[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] += updated_Fsys 

  def _inverseGradient(self):
    """ cv == cartesian so return 1"""
    return 1

  def _Jacobian(self):
    """ cv == cartesian -> ln|J| = 0 so return 0"""
    return 0

  def _entropicCorrection(self):
    return self.p["kb"] * self.p["temperature"] * self._Jacobian()

  def _calBiasingForce(self, coord_x, coord_y, d):

    if (self.p["abfCheckFlag"] == "yes" and  self.p["nnCheckFlag"] == "no") or (self.p["abfCheckFlag"] == "yes" and  self.p["nnCheckFlag"] == "yes" and self.criteriaFEBool == 0):
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

    if (self.p["abfCheckFlag"] == "yes" and  self.p["nnCheckFlag"] == "yes") and self.criteriaFEBool > 0:
      if self.p["ndims"] == 1:
        return -self.colvars_force_NN[getIndices(coord_x, self.bins)]
      if self.p["ndims"] == 2:
        return -self.colvars_force_NN[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)]

  def _abfDecorator(func):
    def _wrapper(self, coord_x, d, vel, coord_y):
      Fabf = func(self, coord_x, d, vel, coord_y)
      currentFsys = self.initializeForce.getForce(coord_x, d, vel, coord_y)
      self._forceDistrRecord(coord_x, currentFsys, coord_y, d)
      self._forceHistDistrRecord(coord_x, coord_y, d)
      return Fabf + currentFsys 
    return _wrapper

  @_abfDecorator
  def getCurrentForce(self, coord_x, d, vel, coord_y): 

    if self.p["abfCheckFlag"] == "no" and self.p["nnCheckFlag"] == "no":
      Fabf = 0  
    else: 
      Fabf = self._calBiasingForce(coord_x, coord_y, d)
    return Fabf

  def _criteriaModCurr(self):
    if self.criteriaCounter <= 1:
      self.criteria_prev = copy.deepcopy(self.colvars_hist / np.sum(self.colvars_hist)) # w/o unbiasing
    else:
      self.criteria_curr = copy.deepcopy(self.colvars_hist / np.sum(self.colvars_hist)) # w/o unbiasing

  def _criteriaModPrev(self):
      self.criteria_prev = copy.deepcopy(self.criteria_curr)

  def _criteriaCheck(self, holder, prev, curr, msERROR):
    if self.criteriaCounter >= 2:
      holder = ((prev - curr)/ curr)** 2 
      holder[np.isnan(holder)] = 0.0   
      holder[np.isinf(holder)] = 0.0    
      holder[np.isneginf(holder)] = 0.0  
      holder = holder[holder > msERROR]
      return not holder.size 
    return False

  def _accumulateColvarsHist(self):
    for n in range(self.p["nparticle"]):
      if self.p["ndims"] == 1:
        self.colvars_hist[getIndices(self.current_coord[n][0], self.bins)] += 1
      elif self.p["ndims"] == 2:
        self.colvars_hist[getIndices(self.current_coord[n][0], self.bins)][getIndices(self.current_coord[n][1], self.bins)] += 1 

  def getCurrentFreeEnergy(self):

    if self.p["nnCheckFlag"] == "no":
      self.colvars_force_tmp = copy.deepcopy(self.colvars_force / self.colvars_count)
      self.colvars_force_tmp[np.isnan(self.colvars_force_tmp)] = 0 
      self.colvars_force_tmp = paddingRightMostBin(self.colvars_force_tmp)
    else:
      self.colvars_force_tmp = copy.deepcopy(self.colvars_force_NN)
      self.colvars_force_tmp[np.isnan(self.colvars_force_tmp)] = 0 
      self.colvars_force_tmp = paddingRightMostBin(self.colvars_force_tmp)
    
    self.colvars_FreeE = integrator(self.p["ndims"], self.colvars_coord, self.colvars_force_tmp, self.p["half_boxboundary"], self.p["init_frame"], self.p["shiftConst"], "tempFreeE.dat")

  def _learningProxy(self):
    if self.p["nnCheckFlag"] == "yes":
        output = trainingANN("loss.dat", "hyperparam.dat", self.p["ndims"], len(self.bins)) 

        self.colvars_force = (self.colvars_force / self.colvars_count)
        self.colvars_force[np.isnan(self.colvars_force)] = 0 # 0/0 = nan n/0 = inf
        self.colvars_force = paddingRightMostBin(self.colvars_force)

        if self.p["init_frame"] < self.p["earlyStopCheck"] * self.p["switchSteps"]:
          self.colvars_force_NN = \
          output.training(self.colvars_coord, self.colvars_force, self.p["earlyLearningRate"], self.p["earlyRegularCoeff"], self.p["earlyEpoch"], self.p["nnOutputFreq"]) 
        else:
          self.colvars_force_NN = \
          output.training(self.colvars_coord, self.colvars_force, self.p["lateLearningRate"], self.p["lateRegularCoeff"], self.p["lateEpoch"], self.p["nnOutputFreq"]) 
  
        self.colvars_force = (self.colvars_force * self.colvars_count)

  def mdrun(self):

    init_real_world_time = time.time()

    # PRE-PROCESSING
    lammpstrj      = open("m%.1f_T%.3f_gamma%.4f_len_%d_%s_%s.lammpstrj" % (self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"], self.p["abfCheckFlag"], self.p["nnCheckFlag"]), "w")
    forceOnCVs     = open("Force_m%.1fT%.3f_gamma%.4f_len_%d_%s_%s.dat"  % (self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"], self.p["abfCheckFlag"], self.p["nnCheckFlag"]), "w")
    histogramOnCVs = open("Hist_m%.1fT%.3f_gamma%.4f_len_%d_%s_%s.dat"   % (self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"], self.p["abfCheckFlag"], self.p["nnCheckFlag"]), "w")

    withANN        = open("instantForceWANN_"  + str(self.p["ndims"]) + "D.dat", "a")
    woANN          = open("instantForceWOANN_" + str(self.p["ndims"]) + "D.dat", "a")
    earlyFreeE     = open("earlyFreeE.dat", "a")
    FinalFE        = open("FreeE.dat", "w")

    # START of the simulation
    self.IO.writeParams(self.p)
    self.IO.lammpsFormatColvarsOutput(self.p["ndims"], self.p["nparticle"], self.p["half_boxboundary"], self.p["init_frame"], self.current_coord, lammpstrj, self.p["writeFreq"]) 
    self.IO.printCurrentStatus(self.p["init_frame"], init_real_world_time)  
    
    while self.p["init_frame"] < self.p["total_frame"]: 

      self.p["init_frame"] += 1
      self.IO.printCurrentStatus(self.p["init_frame"], init_real_world_time)  
      self.mdInitializer.checkTargetTemperature(self.current_vel, self.p["init_frame"], self.p["total_frame"])

      if self.p["init_frame"] % self.p["earlyStopCheck"] == 0 and self.p["init_frame"] != 0 and self.p["abfCheckFlag"] == "yes":
        self.criteriaCounter += 1 
        self._criteriaModCurr()

        if self._criteriaCheck(self.criteria_hist, self.criteria_prev, self.criteria_curr, self.p["trainingCriteria"]):

          if self.p["nnCheckFlag"] == "no": 
            self.getCurrentFreeEnergy() 
            self.IO.certainFrequencyOutput(self.colvars_coord, self.colvars_FreeE, self.colvars_hist, self.p["init_frame"], earlyFreeE)

          else:
            self._learningProxy()
            self.getCurrentFreeEnergy() 
            self.IO.certainFrequencyOutput(self.colvars_coord, self.colvars_FreeE, self.colvars_hist, self.p["init_frame"], earlyFreeE)

          self._criteriaModPrev()

          # retrieve FE
          if self.criteriaFEBool % 2 == 0:
              self.colvars_FreeE_prev = copy.deepcopy(self.colvars_FreeE)
          else:
              self.colvars_FreeE_curr = copy.deepcopy(self.colvars_FreeE)

          self.criteriaFEBool += 1
          
          # To clean invalid data (inf, nan), we let some data == 0, this would cause issues when evaluating criteria, so we should at least do twice
          if self._criteriaCheck(self.criteria_FreeE, self.colvars_FreeE_prev, self.colvars_FreeE_curr, self.p["simlEndCriteria"]) and self.criteriaFEBool >= 2:
            break
          # retrieve FE

      self.mdInitializer.velocityVerletSimple(self.current_coord, self.current_vel) 
      self._accumulateColvarsHist()

      self.IO.lammpsFormatColvarsOutput(self.p["ndims"], self.p["nparticle"], self.p["half_boxboundary"], self.p["init_frame"], self.current_coord, lammpstrj, self.p["writeFreq"]) 
    # END of the simulation

    # POST-PROCESSING
    probability = copy.deepcopy((self.colvars_hist / np.sum(self.colvars_hist)))
    probability = paddingRightMostBin(probability)
    self.IO.propertyOnColvarsOutput(self.bins, probability, self.colvars_hist, histogramOnCVs)
    self.IO.propertyOnColvarsOutput(self.bins, self.colvars_FreeE, self.colvars_count, FinalFE)

    if self.p["nnCheckFlag"] == "yes":
      self.IO.propertyOnColvarsOutput(self.bins, self.colvars_force_NN, self.colvars_count, forceOnCVs)
      self.IO.propertyOnColvarsOutput(self.bins, self.colvars_FreeE, self.colvars_count, forceOnCVs)

    else:
      self.colvars_force = (self.colvars_force / self.colvars_count)
      self.colvars_force[np.isnan(self.colvars_force)] = 0
      self.colvars_force = paddingRightMostBin(self.colvars_force) 
      self.IO.propertyOnColvarsOutput(self.bins, self.colvars_force, self.colvars_count, forceOnCVs)

    # ndims >= 2 -> plot using matplotlib 
    if self.p["ndims"] == 2: 
      s = rendering(self.p["ndims"], self.p["half_boxboundary"], self.p["binNum"], self.p["temperature"])
      s.render(probability, name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "boltzDist" +str(self.p["ndims"])+"D"))
      s.render(self.colvars_FreeE, name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "FEsurface" +str(self.p["ndims"])+"D"))
      if self.p["nnCheckFlag"] == "yes":
        try:
          s.render(self.colvars_force_NN[0], name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "forcex" +str(self.p["ndims"])+"D"))
          s.render(self.colvars_force_NN[1], name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "forcey" +str(self.p["ndims"])+"D"))
        except: # no chance to train 
          pass
      else:
        s.render(self.colvars_force[0],    name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "forcex" +str(self.p["ndims"])+"D"))
        s.render(self.colvars_force[1],    name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "forcey" +str(self.p["ndims"])+"D"))

    # Close files, mkdir and mv files
    self.IO.closeAllFiles(lammpstrj, forceOnCVs, histogramOnCVs, withANN, woANN, earlyFreeE, FinalFE)
    self.IO.makeDirAndMoveFiles(self.p["ndims"], self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"],\
                                self.p["abfCheckFlag"], self.p["nnCheckFlag"], __class__.__name__)

if __name__ == "__main__":
  ABF("in.ABF_1D").mdrun()
  #ABF("in.ABF_2D").mdrun()
