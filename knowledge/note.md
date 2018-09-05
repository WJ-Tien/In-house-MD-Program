# https://www.researchgate.net/post/Whats_the_difference_between_Brownian_dynamics_and_molecular_dynamics
# http://highscope.ch.ntu.edu.tw/wordpress/?p=59534&cpage=1 （all about damping)
# https://www.weizmann.ac.il/sb/faculty_pages/Levy/sites/weizmann.ac.il.sb.faculty_pages.Levy/files/Molecular_dynamics_simulation.pdf
# https://www.hindawi.com/journals/bmri/2015/183918/
# http://csg.sph.umich.edu/abecasis/class/2006/615.19.pdf (simulation annealing)

# TMP Chem youtbe and github

* Brownian dynamics doesn't update the velocities of particles and only considers the changes of positions.
* Brownian dynamics is a simplified form of Langevin dynamics
* Basically, in the equation of the motion of a particle using brownian dynamics, it is considered the hydrodynamic terms for the motion of a particle in a fluid, and in such equation, it is vanished the inertial terms of the motion associated to the particle. In comparison, the Langevin dynamics takes into account the inertial terms for resolving the equation of motion of a particle embedded in a fluid.

* Using molecular dynamics implies solving Newton equations of motion
for many-particle system. E.g., the system configuration
at a certain time point is obtained as the result of solving
system of ordinary differential equations of second order,
for coordinates and impulses.
Brownian dynamics (which is oftenly referred to as Langevin dynamics as well)
deals with systems embedded in thermostat. This means that, along with
regular interparticle potential forces, two additional type of forces
acting on particles have to be taken into account, the friction and stochastic
(Langevin) forces. In the case that the observation time is long or
the friction is strong, the Langevin equations reduce to the overdamped case
where only the coordinates of particles are present. In this limiting case
the inertial forces are neglected and
it is assumed that the velocity distribution is Maxwellian and the averaging
over velocities is already performed. This overdamped case is also referred to
as Brownian dynamics. In thermal equilibrium, the results obtained within these
3 approaches should be identical. For nonequilibrium processes,
BD describes the relaxation of as system embedded in thermostat to thermal equilibrium.

* Although the information given in two previous answers is overall correct, there is a little of confusion about how molecular, Langevin and Brownian dynamics relate to each other. In fact, Langevin dynamics provide a general theoretical frame to two others. The Langevin equation of motion consists of three terms: 1) forces arising from interactions with other particles, 2) forces arising from momenta inherited from the previous time step (this term contains a damping multiplier that accounts for viscosity of the medium), and 3) random forces. All three terms might be useful in molecular dynamics, because MD is not only about moving particles of interest, but also maintaining constant temperature and pressure or imitating a solvent. Then, if in term (2) the damping constant would be large enough, then momenta from the previous step (inertia) would not contribute to the motion anymore. This special case would be called overdamped Langevin dynamics or Brownian dynamics.

過阻尼：因為阻尼很大、緩衝足夠，整個系統有足夠大的能力去將能量耗散，所以在回到平衡的過程中不會衝過頭，系統不會來回振動，而是單調地、緩緩地趨於平衡位置。(系統只能緩緩地趨於平衡位置，原因是大部分的回復力都被拿去克服阻尼。)

欠阻尼：則是阻尼小、耗散慢，所以回復力把系統拉回平衡位置後還會衝過頭，來回多做幾次週期振盪，但隨時間增加，振盪的振幅會越來越小。

臨界阻尼：它阻撓運動的能力是介在上述兩種情況中間，具有防止振盪所需的最低能力。

============================================================================================================================================================================

Ensemble – collection of all possible
systems which have different microscopic
states but identical macroscopic or
thermodynamic state

#ensemble average is hard to derive --> time average

Need to consider the average since the sample 
contains a large number of conformations
 Average values in statistical mechanics corresponds
to ensemble averages
 Calculating this requires integrating over all possible
states of the system which is extremely difficult
 Ergodic
hypothesis states that the time averages
equal the ensemble average
‹A› ensem
ble = ‹A› time
The basic idea is to allow the system to evolve in time
so that the system will eventually pass through all
possible states

van der waals
Express the interaction energy
between two atoms
 Contains an attractive part and a
repulsive part
 Attractive forces due to London
forces (dipole –
dipole interaction)
 Repulsive part due to Pauli-exclusion
principle and inter-nuclear repulsion
 A,C parameters can be determined
by a variety of methods 

tau (tau T or tau P) is a coupling parameter which determines
how tightly the bath and the system are
coupled together. Large means weak
coupling and vice versa.
In order to alter the temperature the
velocities need to be scaled.

####Molecular Dynamics Steps####
#####Initialization ---> Equilibrium ---> Production####

# Initialization 
Specify the initial coordinates and velocities
 Initial structure is obtained by x-ray
crystallography or NMR structure
 Velocities are chosen randomly from
Maxwll-Bolzmann or Gaussian
distribution:


* Energy Minimization (e.g. steepest decent method, newton raphson (rare))
Prior to starting a simulation it is
advisable to do energy minimization
 Useful in correcting flaws such as steric
clashes between atoms and distorted
bond angles/lengths
 Need to find the minimum value for the
energy function with 3N degrees of
freedom

* Solvation
The solvent (usually water) has a fundamental
influence on the structure, dynamics and
thermodynamics of biological molecules
 One of the most important effects of the
solvent is the screening of electrostatic
interactions
 There are several methods for the treatment of
the solvent effect:
z implicit treatment of the solvent: an effective
dielectric constant is used. Often the effective
dielectric constant is taken to be distance dependent
z Periodic boundary conditions

* Periodic Boundary Condition

Periodic boundary
conditions enable a
simulation to be
performed using a
relatively small number of
particles in such a way
that the particles
experience forces as
though they were in a
bulk solution
 The box containing the
system is replicated
infinitely in all directions
to give a periodic array

#Equilibrium Phase
During this phase several properties
are monitored: pressure, temperature
and energy
The purpose of this phase is to run the
simulation until these properties
become stable with time

#Production run 
calculate thermodynamics properties of the system

======================================================

#####Langevin dynamics####
The Langevin equation is a stochastic
differential equation in which two force
terms have been added to Newton's second law
A molecule in the real world is not present
in a vacuum. Jostling of solvent molecules
causes friction and there are also collisions
that perturb the system
The model was employed to avoid explicit
representation of water molecules, enhance
sampling, represent hydration shell models
in large system and other uses.

The effects of solvent molecules can be
approximated in terms of a frictional drag
on the solute as well as random kicks
associated with the thermal motions of the
solvent molecules
The molecules interact with a stochastic
heat bath via random forces and dissipative
forces 


γ represent the collision frequency: controls the
magnitude of the frictional force and the variance
of the random forces.
it ensures that the system converges to Bolzmann
distribution.
The larger its value, the greater the influence of
the surrounding fluctuating force (solvent). Small
implies inertial motion.
If the main objective is to control temperature, one
need to use small values of γ

3 different situations can be considered,
depending on the magnitudes of the
integration time step and :
1. γ*dt <= 1 : small influence of the solvent
2. γ*dt >= 1: diffusion limit
3. Intermediate
for each of those limits can find algorithms to
solve the differential equations.
R(t) = random force vector (staionary gaussian process with zero mean)


#advantage of LD
The number of degrees of freedom is the
number of solute which leads to a reduction
in time of calculations. This allows to perform
longer simulations
{ The random motions can assist in
propagating barrier-crossing motions and
therefore can improve conformational
sampling characteristics
{ Efficient in simulations of long chain
molecules such as DNA with many base
pairs

#Brownian Dynamics 
Describes the diffusive limit of the
langevin equation where the motion is
more random
{ Assumes a high viscosity of the fluid
such that the solute reorients
continuously by colliding with the
solvent molecules
{ Uses a very large friction coefficient
{ Useful in describing processes such as
diffusion controlled reactions

#Monte Carlo
Generates configurations of the system by
making random changes
{ Algorithm: calculate the energy of the new
configuration
z if ∆E<0 then the configuration is accepted
z if ∆E>0 then it will depend on Bolzmann
factor.
This process is iterated until sufficient
sampling statistics for the current
temperature are achieved

#Solvent Effects
Two options for taking solvent into account
– Explicitly represent solvent molecules
• High computational expense but more accurate
• Usually assume periodic boundary conditions (a water
molecule that goes off the left side of the simulation box will
come back in the right side, like in PacMan)
– Implicit solvent
• Mathematical model to approximate average effects of
solvent
• Less accurate but faster 


#Q&A
Why MD and MC can generate (almost)  the same distribution

--> exp(-B(K+U)) K=3/2 * (kBT)
---> Const * exp(-BU) for MD; exp(-BU) for MC

#Q&A
why random number v < boltzmann factor in MC
we cannot just ignore the higher energy term (In real world, the state with higher energy might occur)
boltzmann factor ~= probability 

#Monte Carlo and Metropolis monte carlo
Low probability term contribute to the system properties the most
Monte Carlo: equal contribution of data points in the phase space --> low efficiency --> most structures does not contribute --->  (exp(-BH)) H high --> low prob --> low contribution
Metropolis Monte Carlo: focus on the low prob  (MC method biases to low E) (if Ei >> kBT --> prob ~=0) ---> jagged path --> no time(just trials) --> adjust opt_scalar to 50%  accpetance ratio
--> multiple local minimim issues

	

