# Lattice QED Hamiltonian for Quantum Computing

'HC_Lattice.py'
A python code that builds a lattice in generic N dimensions, with periodic or open boundary conditions.
It finds the set of links and sites, build plaquettes and chain for Jordan_Wigner definition (version up to 3D).

'Hamiltonian_QED,py' (draft)
A python code that builds Hamiltonian N-dimensional lattice, both with open and periodic boundary conditions. The formulation considered is from Kogut and Susskind and the Gauss’ law is applied.
By doing so, we will get a gauge invariant system, reduce the number of dynamical links needed
for the computation, and have a more resource-efficient Hamiltonian suitable for a wide range of
quantum hardware.

'Hamiltonianqed.py':
A python code that builds Hamiltonian on a generic N × N lattice, both with open and periodic boundary conditions. The formulation and methods like the other case, only this is limited to 1D and 2D lattices.