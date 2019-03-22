#!/usr/bin/env python3
from subprocess import Popen
import numpy as np
import time

class mdFileIO(object):

  def __init__(self):

    self.params = {}
  
  def readParamFile(self, inputFile):

    """ read .mdp files with format param_names = value """

    with open(inputFile, "r") as fin:

      for line in fin:

        if len(line.strip()) != 0: # ignore empty lines

          if line[0] == ";": # ignore comments
            pass

          else:
            line = line.split()

            if line[1] != "=":
              print("check the format in .mdp -> the format must be param_names = value")
              exit(1)

            if line[0] == "kb":
              self.params["kb"] = int(line[2])

            elif line[0] == "ndims":
              self.params["ndims"] = int(line[2])

            elif line[0] == "mass":
              self.params["mass"] = float(line[2])
              
            elif line[0] == "temperature":
              self.params["temperature"] = float(line[2])
              
            elif line[0] == "frictCoeff":
              self.params["frictCoeff"] = float(line[2])

            elif line[0] == "writeFreq":
              self.params["writeFreq"] = int(line[2])
            
            elif line[0] == "earlyLearningRate":
              self.params["earlyLearningRate"] = float(line[2])
            
            elif line[0] == "earlyEpoch":
              self.params["earlyEpoch"] = int(line[2])

            elif line[0] == "earlyRegularCoeff":
              self.params["earlyRegularCoeff"] = float(line[2])
    
            elif line[0] == "switchSteps":
              self.params["switchSteps"] = int(line[2])

            elif line[0] == "trainingFreq":
              self.params["trainingFreq"] = int(line[2])

            elif line[0] == "certainOutFreq":
              self.params["certainOutFreq"] = int(line[2])

            elif line[0] == "lateLearningRate":
              self.params["lateLearningRate"] = float(line[2])

            elif line[0] == "lateEpoch":
              self.params["lateEpoch"] = int(line[2])

            elif line[0] == "lateRegularCoeff":
              self.params["lateRegularCoeff"] = float(line[2])

            elif line[0] == "half_boxboundary":
              self.params["half_boxboundary"] = float(line[2])

            elif line[0] == "binNum":
              self.params["binNum"] = int(line[2])

            elif line[0] == "nparticle":
              self.params["nparticle"] = int(line[2])

            elif line[0] == "init_time":
              self.params["init_time"] = float(line[2])

            elif line[0] == "time_step":
              self.params["time_step"] = float(line[2])

            elif line[0] == "init_frame":
              self.params["init_frame"] = int(line[2])

            elif line[0] == "total_frame":
              self.params["total_frame"] = int(line[2])

            elif line[0] == "thermoStatFlag":
              self.params["thermoStatFlag"] = line[2] 

            elif line[0] == "nnOutputFreq":
              self.params["nnOutputFreq"] = int(line[2])

            elif line[0] == "abfCheckFlag":
              self.params["abfCheckFlag"] = line[2]

            elif line[0] == "nnCheckFlag":
              self.params["nnCheckFlag"] = line[2]

            else:
              print("unknown argument -> %s" % (line[0]))
              exit(1)
        else: # do nothing with empty lines 
          pass  
          
    self.params["box"] = np.ones(self.params["ndims"]) * self.params["half_boxboundary"] * 2
      
    return self.params

  def writeParams(self, params):

    with open("simulation_params.dat", "w") as fout:
      fout.write("#" + " " + "thermoStatFlag" + " " + str(params["thermoStatFlag"]) + "\n")
      fout.write("#" + " " + "ndim"           + " " + str(params["ndims"])          + "\n")
      fout.write("#" + " " + "nparticle"      + " " + str(params["nparticle"])      + "\n")
      fout.write("#" + " " + "binNumber"      + " " + str(params["binNum"])         + "\n")
      fout.write("#" + " " + "temperature"    + " " + str(params["temperature"])    + "\n") 
      fout.write("#" + " " + "mass"           + " " + str(params["mass"])           + "\n") 
      fout.write("#" + " " + "frictCoeff"     + " " + str(params["frictCoeff"])     + "\n") 
      fout.write("#" + " " + "total_frame"    + " " + str(params["total_frame"])    + "\n") 
      fout.write("#" + " " + "time_step"      + " " + str(params["time_step"])      + "\n") 
      fout.write("#" + " " + "abfCheckFlag"   + " " + str(params["abfCheckFlag"])   + "\n")
      fout.write("#" + " " + "nnCheckFlag"    + " " + str(params["nnCheckFlag"])    + "\n")
      fout.write("#" + " " + "certainOutFreq" + " " + str(params["certainOutFreq"]) + "\n")

  def _lammpsFileHeader(self, nparticle, half_boxboundary, frame, lammpstrj):

      lammpstrj.write("ITEM: TIMESTEP" + "\n")  
      lammpstrj.write(str(frame) + "\n")
      lammpstrj.write("ITEM: NUMBER OF ATOMS" + "\n")
      lammpstrj.write(str(nparticle) + "\n")
      lammpstrj.write("ITEM: BOX BOUNDS pp pp pp" + "\n")
      lammpstrj.write("%-.3f %.3f\n" % (-half_boxboundary, half_boxboundary))
      lammpstrj.write("%-.3f %.3f\n" % (-half_boxboundary, half_boxboundary))
      lammpstrj.write("%-.3f %.3f\n" % (-half_boxboundary, half_boxboundary))
      lammpstrj.write("ITEM: ATOMS id type x y z" + "\n") 

  def lammpsFormatColvarsOutput(self, ndims, nparticle, half_boxboundary, frame, coord, lammpstrj, outputfreq):
      #TODO id number of the atom
      if frame % outputfreq == 0:
        self._lammpsFileHeader(nparticle, half_boxboundary, frame, lammpstrj)

        for i in range(nparticle):

          if ndims == 1:
            lammpstrj.write(str(i) + " " + str(1) + " " + str(coord[i][0]) + " " + str(0)           + " " + str(0) + "\n") 

          if ndims == 2:
            lammpstrj.write(str(i) + " " + str(1) + " " + str(coord[i][0]) + " " + str(coord[i][1]) + " " + str(0) + "\n") 

  def _pdbFileHeader(self): #TODO
    pass

  def pdbFormatColvarsOutput(self, coord): #TODO
    #"%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f"
    pass

  def propertyOnColvarsOutput(self, colvars_coord, colvars_property, colvars_count, fileOutProperty): 
    """ reduceDim is dirty and should be replaced in the future """

    colvars_coord = np.array(colvars_coord)
    colvars_property = np.array(colvars_property)
    colvars_count = np.array(colvars_count)

    if len(colvars_property.shape) == 1: # 1D
      for i in range(len(colvars_coord)): 
        fileOutProperty.write(str(colvars_coord[i]) + " ")
        fileOutProperty.write(str(colvars_property[i]) + " " + str(colvars_count[i]) + "\n")  

    elif len(colvars_property.shape) == 2: # 1D
      for i in range(len(colvars_coord)):
        for j in range(len(colvars_coord)):
          fileOutProperty.write(str(colvars_coord[i]) + " ")
          fileOutProperty.write(str(colvars_coord[j]) + " ")
          fileOutProperty.write(str(colvars_property[i][j]) + " " + str(colvars_count[i][j]) + "\n")  

    elif len(colvars_property.shape) == 3: # 1D
      for i in range(len(colvars_coord)):
        for j in range(len(colvars_coord)):
          fileOutProperty.write(str(colvars_coord[i]) + " ")
          fileOutProperty.write(str(colvars_coord[j]) + " ")
          fileOutProperty.write(str(colvars_property[0][i][j]) + " " + str(colvars_count[0][i][j]) + " " +str(colvars_property[1][i][j]) + " " + str(colvars_count[1][i][j]) + "\n")  
  
  def certainFrequencyOutput(self, colvars_coord, colvars_property, colvars_count, frame, outputFreq, fileOutProperty): #TODO

    colvars_coord = np.array(colvars_coord)
    colvars_property = np.array(colvars_property)
    colvars_count = np.array(colvars_count)

    if frame % outputFreq == 0:
      fileOutProperty.write("# " + str(frame) + "\n")

      if len(colvars_property.shape) == 1: # 1D
        for i in range(len(colvars_coord)): 
          fileOutProperty.write(str(colvars_coord[i]) + " ")
          fileOutProperty.write(str(colvars_property[i]) + " " + str(colvars_count[i]) + "\n")  
        fileOutProperty.write("\n")


      elif len(colvars_property.shape) == 2: # 2D
        for i in range(len(colvars_coord)):
          for j in range(len(colvars_coord)):
            fileOutProperty.write(str(colvars_coord[i]) + " ")
            fileOutProperty.write(str(colvars_coord[j]) + " ")
            fileOutProperty.write(str(colvars_property[i][j]) + " " + str(colvars_count[i][j]) + "\n")  

      elif len(colvars_property.shape) == 3:  # 3D
        for i in range(len(colvars_coord)):
          for j in range(len(colvars_coord)):
            fileOutProperty.write(str(colvars_coord[i]) + " ")
            fileOutProperty.write(str(colvars_coord[j]) + " ")
            fileOutProperty.write(str(colvars_property[0][i][j]) + " " + str(colvars_count[0][i][j]) + " " +str(colvars_property[1][i][j]) + " " + str(colvars_count[1][i][j]) + "\n")  
        fileOutProperty.write("\n")
  
  def makeDirAndMoveFiles(self, ndims, mass, temperature, frictCoeff, total_frame, abfCheckFlag, nnCheckFlag, moduleName):

    dirName = "%dD_m%.2f_T%.3f_g%.4f_len%d_%s_%s_%s" % (ndims, mass, temperature, frictCoeff, total_frame, abfCheckFlag, nnCheckFlag, moduleName)
    makeDirString = "mkdir" + " " + dirName 
    makeDir = Popen(makeDirString, shell=True)
    makeDir.wait()
    mvFileString = "mv" + " " + "*.dat *.lammpstrj *.png" + " " + dirName + " "  
    mvFile = Popen(mvFileString, shell=True) 
    mvFile.wait()
        

  def printCurrentStatus(self, frame, init_real_world_time):
    print("Frame %d with Time %f" % (frame, time.time() - init_real_world_time))

  def closeAllFiles(self, *files):
    for f in files:
      f.close()

if __name__ == "__main__":
  a = mdFileIO()
  #b = a.readParamFile("in.mdp")
  b = np.zeros((1, 1))
  print(b)
  f = open("test.lammpstrj", "w")
  a.lammpsFormatColvarsOutput(1, 1, 3, 0, b, f) 
  f.close()
  #a.writeParams(b)
  #print(len(a))
  #print(a)
  
