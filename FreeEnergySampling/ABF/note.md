## Class importanceSampling (python3 version= testing code) 
1. Method "withFiction_sysPotential" is on the basis of "sysPotential"
2. Method "ABF" are on the basis of "sysForce", "Jacobian", "inverseGradient"

## Nomenclature
1. LDABF = Langevin Dynamics Adaptive Bias Force (Method)
2. bzdistCheck = boltzmann distribution check
3. gamma = friction coefficient
4. TL = time length (simulation time length)
5. temp = temperature
6. w = with; wo = without
7. k = spring Constant

## Note 
1. Dimensionless kb = 1 
2. Higher gamma, better sampling
3. The point on pi must be discarded due to PBC condition (Boltzmann Distribution)
4. For eABF, higher k(springConst), better coupling between Cartcoord and ficcoord 

## Equipartition theorem
1. For three dimensions molecular dynamics, the kinetic energy equals to 3/2 * kb * T
2. For one dimnesion molecular dynamics, the kinetic energy equals to 1/2 * kb * T
