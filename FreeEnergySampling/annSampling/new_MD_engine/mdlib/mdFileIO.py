#!/usr/bin/env python3
import numpy as np

#"%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f"

class mdFileIO(object):

	def __init__(self):
		self.params = {}
	
	def readParamFile(self, inputFile):

		""" read md parameters with format param names = value """

		with open(inputFile, "r") as fin:

			for line in fin:

				if len(line.strip()) != 0: # not empty

					if line[0] == ";":
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
						
						elif line[0] == "earlyLearningRate":
							self.params["learningRate"] = float(line[2])
						
						elif line[0] == "earlyEpoch":
							self.params["epoch"] = int(line[2])

						elif line[0] == "earlyRegularCoeff":
							self.params["regularCoeff"] = float(line[2])
		
						elif line[0] == "switchSteps":
							self.params["switchSteps"] = int(line[2])

						elif line[0] == "trainingFreq":
							self.params["trainingFreq"] = int(line[2])

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

						elif line[0] == "time_length":
							self.params["time_length"] = float(line[2])

						elif line[0] == "abfCheckFlag":
							self.params["abfCheckFlag"] = line[2]

						elif line[0] == "nnCheckFlag":
							self.params["nnCheckFlag"] = line[2]

						else:
							print("unknown arguments -> %s" % (line[0]))
							exit(1)
				else:
					continue	
					
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
			fout.write("#" + " " + "time_length"    + " " + str(params["time_length"])    + "\n") 
			fout.write("#" + " " + "time_step"      + " " + str(params["time_step"])      + "\n") 
			fout.write("#" + " " + "abfCheckFlag"   + " " + str(params["abfCheckFlag"])   + "\n")
			fout.write("#" + " " + "nnCheckFlag"    + " " + str(params["nnCheckFlag"])    + "\n")

	def propertyOnColvarsOutput(self, outputFile):
		pass

	def pdbFormatColvarsOutput(self, coord):
		pass
	
	def certainFrequencyOutput(self, coord, specificProperty):
		pass

	def closeAllFiles(self, *files):
		for f in files:
			f.close()
			
if __name__ == "__main__":
	a = mdFileIO()
	b = a.readParamFile("in.mdp")
	a.writeParams(b)
	#print(len(a))
	#print(a)
	
