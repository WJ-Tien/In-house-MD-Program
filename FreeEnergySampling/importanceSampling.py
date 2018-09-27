#!/usr/bin/env python3
import numpy as np 

class importanceSampling(object):

	def __init__(self):
		self.kb = 1 

	def sysPotential(self, CartesianCoord):
		return np.cos(CartesianCoord) + np.cos(2*CartesianCoord) + np.cos(3*CartesianCoord)

	def sysForce(self, CartesianCoord):
		return np.sin(CartesianCoord) + 2*np.sin(2*CartesianCoord) + 3*np.sin(3*CartesianCoord)

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
		
		beta = 1 / self.kb / temperature
		random_theta = np.random.normal(0, 1)		
		random_xi = np.random.normal(0, 1)		
		sigma = np.sqrt(2.0 * self.kb * temperature * frictCoeff / mass)

		current_force = self.sysForce(current_disp) + (-self.sysForce(current_disp) - \
                    (1 / beta * self.Jacobian())) * self.inverseGradient()
		
		Ct = (0.5 * tintv * tintv * (current_force / mass - frictCoeff * current_vel)) + \
         sigma * (tintv**1.5) * (0.5 * random_xi + 0.288675 * random_theta) 

		next_disp = current_disp + tintv * current_vel + Ct
		next_disp -= (round(next_disp / box_length) * box_length) # PBC

		next_force = self.sysForce(next_disp) + (-self.sysForce(next_disp) - \
                 (1 / beta * self.Jacobian())) * self.inverseGradient()

		next_vel = current_vel + (0.5 * tintv * (next_force + current_force) / mass) - \
							 tintv * frictCoeff * current_vel + sigma * np.sqrt(tintv) * random_xi - \
               frictCoeff * Ct 
			
		next_time = current_time + tintv
		
		MD_PARM[0] = next_disp
		MD_PARM[1] = next_vel
		MD_PARM[2] = next_time

		return MD_PARM

if __name__ == "__main__":

	fout = open("LDABF_gamma_10.dat", "w") 
	MD_PARM = [1.25, 0., 0., 0.005, 1, 6.28319, 10, 4, 20, 0]
	# disp[0], vel[1], time[2], tintv[3], mass[4], boxL[5], frictCoeff[6], temp[7], TimeL[8], initfm[9]
	fout.write(str(MD_PARM[9]) + " " + str(MD_PARM[2]) + " " + str(MD_PARM[0]) + "\n")

	s = importanceSampling()

	while MD_PARM[2] < MD_PARM[8]:
		simResult = s.ABF(MD_PARM[0], MD_PARM[1], MD_PARM[2], MD_PARM[3], MD_PARM[4], MD_PARM[5], MD_PARM[6], MD_PARM[7], MD_PARM)
		MD_PARM[0] = simResult[0]
		MD_PARM[1] = simResult[1]
		MD_PARM[2] = simResult[2]
		MD_PARM[9] += 1	
		fout.write(str(MD_PARM[9]) + " " + str(MD_PARM[2]) + " " + str(MD_PARM[0]) + "\n")

