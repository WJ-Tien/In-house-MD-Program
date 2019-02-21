#!/usr/bin/env python3
import numpy as np

#"%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f"

class mdFileIO(object):
	
	def readParamFile(self, inputFile):

		""" read md parameters with format param names = value """

		params = {}

		with open(inputFile, "r") as fin:

			for line in fin:

				if not line.strip() or line[0] != ";":
					line = line.split()

					if line[1] != "=":
						print("check the format in .mdp -> the format must be param_names = value")
						exit(1)

					if line[0] == "kb":
						params["kb"] = int(line[2])

					elif line[0] == "ndims":
						params["ndims"] = int(line[2])

					elif line[0] == "mass":
						params["mass"] = float(line[2])
						
					elif line[0] == "temperature":
						params["temperature"] = float(line[2])
						
					elif line[0] == "frictCoeff":
						params["frictCoeff"] = float(line[2])
					
					elif line[0] == "learningRate":
						params["learningRate"] = float(line[2])
					
					elif line[0] == "epoch":
						params["epoch"] = int(line[2])

					elif line[0] == "regularCoeff":
						params["regularCoeff"] = float(line[2])
	
					elif line[0] == "switchSteps":
						params["switchSteps"] = int(line[2])

					elif line[0] == "trainingFreq":
						params["trainingFreq"] = int(line[2])

					elif line[0] == "lateLearningRate":
						params["lateLearningRate"] = float(line[2])

					elif line[0] == "lateEpoch":
						params["lateEpoch"] = int(line[2])

					elif line[0] == "lateRegularCoeff":
						params["lateRegularCoeff"] = float(line[2])

					elif line[0] == "half_boxboundary":
						params["half_boxboundary"] = float(line[2])

					elif line[0] == "binNum":
						params["binNum"] = float(line[2])

					elif line[0] == "nparticle":
						params["nparticle"] = int(line[2])

					elif line[0] == "init_time":
						params["init_time"] = float(line[2])

					elif line[0] == "time_step":
						params["time_step"] = float(line[2])

					elif line[0] == "init_frame":
						params["init_frame"] = int(line[2])

					elif line[0] == "mode":
						params["mode"] = line[2] 

					elif line[0] == "nnOutputFreq":
						params["nnOutputFreq"] = int(line[2])

					elif line[0] == "time_length":
						params["time_length"] = float(line[2])

					elif line[0] == "abf_switch":
						params["abf_switch"] = line[2]

					elif line[0] == "nn_switch":
						params["nn_switch"] = line[2]

					else:
						print("unknown arguments -> %s" % (line[0]))
						exit(1)

					
		params["box"] = np.ones(params["ndims"]) * params["half_boxboundary"] * 2
			
		return params

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
	print(mdFileIO().readParamFile("in.mdp"))
	
