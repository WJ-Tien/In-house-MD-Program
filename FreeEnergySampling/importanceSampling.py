#!/usr/bin/env python3
import numpy as np 
import time

class importanceSampling(object):

	def __init__(self):
		self.kb = 1 

	def sysPotential(self, CartesianCoord):
		return np.cos(CartesianCoord) + np.cos(2*CartesianCoord) + np.cos(3*CartesianCoord)

	def sysForce(self, CartesianCoord):
		return np.sin(CartesianCoord) + 2*np.sin(2*CartesianCoord) + 3*np.sin(3*CartesianCoord)

	def withFiction_sysForce(self, CartesianCoord, springConst, fictiousCoord): 
		return springConst * (fictiousCoord - CartesianCoord) 

	def withFiction_sysPotential(self, CartesianCoord, springConst, fictiousCoord): #for eABF
		return self.sysPotential(CartesianCoord) + 0.5 * springConst * (CartesianCoord - fictiousCoord)**2	

	def Jacobian(self):
		# dln|J|/d(xi)
		return 0

	def inverseGradient(self):
		# d(xi)/dx
		return 1

	def ABF(self, current_disp, current_vel, current_time, tintv, mass, box_length, frictCoeff, temperature, MD_PARM):
		
		# Langevin MD
		# x -> xi
		# 1D jacobian = 1
		# inverse gradient partial(xi) / partial(x) = 1
		# partial v(x) / partial(xi) = 1 
		# derivative of Jacobian = 0
		# force act on the colvar xi would average to zero over time
		# temperature here is the target temperature
		# for real temperature T = 2/3/kb * KE
		
		beta = 1 / self.kb / temperature
		random_theta = np.random.normal(0, 1)
		random_xi = np.random.normal(0, 1)
		sigma = np.sqrt(2.0 * self.kb * temperature * frictCoeff / mass)

		current_force = self.sysForce(current_disp) + (-self.sysForce(current_disp) - \
                    (1 / beta * self.Jacobian())) * self.inverseGradient()
#		current_force = self.sysForce(current_disp)	

		Ct = (0.5 * tintv * tintv * (current_force / mass - frictCoeff * current_vel)) + \
         sigma * (tintv**1.5) * (0.5 * random_xi + 0.288675 * random_theta) 

		next_disp = current_disp + tintv * current_vel + Ct
		next_disp -= (round(next_disp / box_length) * box_length) # PBC

		next_force = self.sysForce(next_disp) + (-self.sysForce(next_disp) - \
                 (1 / beta * self.Jacobian())) * self.inverseGradient()
#		next_force = self.sysForce(next_disp)
		next_vel = current_vel + (0.5 * tintv * (next_force + current_force) / mass) - \
							 tintv * frictCoeff * current_vel + sigma * np.sqrt(tintv) * random_xi - \
               frictCoeff * Ct 
			
		next_time = current_time + tintv
		
		MD_PARM[0] = next_disp
		MD_PARM[1] = next_vel
		MD_PARM[2] = next_time

		return MD_PARM

	def eABF(self):
		pass


if __name__ == "__main__":

	startTime = time.time()
	fout = open("LDABF_gamma_1_TL_100000_temp_4_wABF.dat", "w") 
	MD_PARM = [0., 2., 0., 0.005, 1, 6.283185307179586, 1, 4, 100000, 0]
	T_REAL =  MD_PARM[1]**2 *  MD_PARM[4] / 1 # onedim 0.5 * mv^2 = 0.5 * kbT -> equi theorem 
	# disp[0], vel[1], time[2], tintv[3], mass[4], boxL[5], frictCoeff[6], temp[7], TimeL[8], initfm[9]
	fout.write(str(MD_PARM[9]) + " " + str(MD_PARM[2]) + " " + str(round(MD_PARM[0], 6)) + " " + str(T_REAL) + "\n")

	s = importanceSampling()

	while MD_PARM[2] < MD_PARM[8]:
		simResult = s.ABF(MD_PARM[0], MD_PARM[1], MD_PARM[2], MD_PARM[3], MD_PARM[4], MD_PARM[5], MD_PARM[6], MD_PARM[7], MD_PARM)
		MD_PARM[0] = simResult[0]
		MD_PARM[1] = simResult[1]
		MD_PARM[2] = simResult[2]
		MD_PARM[9] += 1	
		print("step %d with time %f " % (MD_PARM[9], time.time() - startTime))
		T_REAL = MD_PARM[1]**2 *  MD_PARM[4] / 1 
		fout.write(str(MD_PARM[9]) + " " + str(MD_PARM[2]) + " " + str(round(MD_PARM[0], 6)) + " " + str(T_REAL) + "\n")
