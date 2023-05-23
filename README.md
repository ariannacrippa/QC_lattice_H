# QC_lattice_H

'Hamiltonianqed.py':
A python code that builds Hamiltonian on a generic N × N lattice, both with open and periodic boundary conditions. The formulation considered is from Kogut and Susskind and the Gauss’ law is applied.
By doing so, we will get a gauge invariant system, reduce the number of dynamical links needed
for the computation, and have a more resource-efficient Hamiltonian suitable for a wide range of
quantum hardware.

'HC_Lattice.py'
A python code that builds a lattice in generic N dimensions, with periodic or open boundary conditions.
It finds the set of links and sites, build plaquettes and chain for Jordan_Wigner definition (version up to 3D).