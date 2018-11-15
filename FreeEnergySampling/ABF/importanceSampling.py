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

	def withFiction_sysForce(self, CartesianCoord, springConst, fictiousCoord):     # for eABF 
		return springConst * (fictiousCoord - CartesianCoord) 

	def withFiction_sysPotential(self, CartesianCoord, springConst, fictiousCoord): # for eABF
		return self.sysPotential(CartesianCoord) + 0.5 * springConst * (CartesianCoord - fictiousCoord)**2	

	def Jacobian(self):        # for conventional ABF
		# dln|J|/d(xi)
		return 0

	def inverseGradient(self): # for conventional ABF
		# d(xi)/dx
		return 1

	def ABF(self, current_disp, current_vel, current_time, tintv, mass, box_length, frictCoeff, temperature, MD_PARM):
		
		# Langevin MD
		# x -> xi; Cartesian Coord -> Collective Variables (reaction coordinates)
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

	def eABF(self, current_disp, current_vel, current_time, tintv, mass, box_length, frictCoeff, temperature, springConst, current_disp_fictitious, current_vel_fictitious, MD_PARM):

		beta = 1 / self.kb / temperature
		random_theta = np.random.normal(0, 1)
		random_xi = np.random.normal(0, 1)
		sigma = np.sqrt(2.0 * self.kb * temperature * frictCoeff / mass)

		current_force = self.sysForce(current_disp) + self.withFiction_sysForce(current_disp, springConst, current_disp_fictitious)
		current_force_fictitious = 0.

		Ct = (0.5 * tintv * tintv * (current_force / mass - frictCoeff * current_vel)) + \
         sigma * (tintv**1.5) * (0.5 * random_xi + 0.288675 * random_theta) 
		Ct_fictitious = (0.5 * tintv * tintv * (current_force_fictitious / mass - frictCoeff * current_vel_fictitious)) + \
         sigma * (tintv**1.5) * (0.5 * random_xi + 0.288675 * random_theta) 

		next_disp = current_disp + tintv * current_vel + Ct
		next_disp -= (round(next_disp / box_length) * box_length) # PBC

		next_disp_fictitious = current_disp_fictitious + tintv * current_vel_fictitious + Ct_fictitious
		next_disp_fictitious -= (round(next_disp_fictitious / box_length) * box_length) # PBC


		next_force = self.sysForce(next_disp) + self.withFiction_sysForce(next_disp, springConst, next_disp_fictitious)
		next_force_fictitious = 0.

		next_vel = current_vel + (0.5 * tintv * (next_force + current_force) / mass) - \
							 tintv * frictCoeff * current_vel + sigma * np.sqrt(tintv) * random_xi - \
               frictCoeff * Ct 
		next_vel_fictitious = current_vel_fictitious + (0.5 * tintv * (next_force_fictitious + current_force_fictitious) / mass) - \
							            tintv * frictCoeff * current_vel_fictitious + sigma * np.sqrt(tintv) * random_xi - \
                          frictCoeff * Ct_fictitious 

		next_time = current_time + tintv
		
		MD_PARM[0] = next_disp
		MD_PARM[1] = next_vel
		MD_PARM[2] = next_time
		MD_PARM[9] = next_disp_fictitious 
		MD_PARM[10] = next_vel_fictitious
	
		return MD_PARM
		
if __name__ == "__main__":

	# The Followings are the testing codes

	'''
	startTime = time.time()
	fout = open("LDABF_gamma_1_TL_100000_temp_10_wABF.dat", "w") 
	MD_PARM = [0.,3.1622776601683795, 0., 0.005, 1, 6.283185307179586, 1, 10, 100000, 0]
	disp[0], vel[1], time[2], tintv[3], mass[4], boxL[5], frictCoeff[6], temp[7], TimeL[8], initfm[9]
	onedim 0.5 * mv^2 = 0.5 * kbT -> equipartition theorem 
	T_REAL =  MD_PARM[1]**2 *  MD_PARM[4] / 1 
	fout.write(str(MD_PARM[9]) + " " + str(MD_PARM[2]) + " " + str(round(MD_PARM[0], 6)) + " " + str(T_REAL) + "\n")
	
	while MD_PARM[2] < MD_PARM[8]:
		simResult = s.ABF(MD_PARM[0], MD_PARM[1], MD_PARM[2], MD_PARM[3], MD_PARM[4], MD_PARM[5], MD_PARM[6], MD_PARM[7], MD_PARM)
		MD_PARM[0] = simResult[0]
		MD_PARM[1] = simResult[1]
		MD_PARM[2] = simResult[2]
		MD_PARM[9] += 1	
		print("step %d with time %f " % (MD_PARM[9], time.time() - startTime))
		T_REAL = MD_PARM[1]**2 *  MD_PARM[4] / 1 
		fout.write(str(MD_PARM[9]) + " " + str(MD_PARM[2]) + " " + str(round(MD_PARM[0], 6)) + " " + str(T_REAL) + "\n")

'''

	startTime = time.time()
	fout = open("LDeABF_v2_gamma_1_TL_100000_temp_4_k_10000_weABF.dat", "w") 
	MD_PARM = [0.,3.1622776601683795, 0., 0.005, 1, 6.283185307179586, 1, 4, 1000, 0., 3.1622776601683795, 100000, 0]
	fout.write(str(MD_PARM[12]) + " " + str(MD_PARM[2]) + " " + str(round(MD_PARM[0], 6)) + " " + str(round(MD_PARM[9], 6)) + "\n")
	# disp[0], vel[1], time[2], tintv[3], mass[4], boxL[5], frictCoeff[6], temp[7], springConst[8], fic_disp[9], fic_vel[10], TimeL[11], inifm[12] 

	s = importanceSampling()

	while MD_PARM[2] < MD_PARM[11]:
		simResult = s.eABF(MD_PARM[0], MD_PARM[1], MD_PARM[2], MD_PARM[3], MD_PARM[4], MD_PARM[5], MD_PARM[6], MD_PARM[7], MD_PARM[8], MD_PARM[9], MD_PARM[10], MD_PARM)
		MD_PARM[0] = simResult[0]
		MD_PARM[1] = simResult[1]
		MD_PARM[2] = simResult[2]
		MD_PARM[9] = simResult[9]
		MD_PARM[10] = simResult[10]
		MD_PARM[12] += 1	
		print("step %d with time %f " % (MD_PARM[12], time.time() - startTime))
		fout.write(str(MD_PARM[12]) + " " + str(MD_PARM[2]) + " " + str(round(MD_PARM[0], 6)) + " " + str(round(MD_PARM[9], 6)) + "\n")

