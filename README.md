# Lattice QED Hamiltonian for Quantum Computing

In this repository one can write the QED Hamiltonian for a 1,2 or 3D lattice. Compatible with exact diagonalisation and Qiskit library.

'HC_Lattice.py'
A python code that builds a lattice in generic N dimensions, with periodic or open boundary conditions.
It finds the set of links and sites, build plaquettes and chain for Jordan_Wigner definition (version up to 3D).

'Hamiltonian_QED_sym.py'
A python code that builds a symbolic expression of QED Hamiltonian N-dimensional lattice, both with open and periodic boundary conditions. The formulation considered is from Kogut and Susskind and the Gauss’ law is applied.
By doing so, we will get a gauge invariant system, reduce the number of dynamical links needed
for the computation, and have a more resource-efficient Hamiltonian suitable for a wide range of
quantum hardware.

'Hamiltonian_QED_oprt.py'
A python code that imports the Hamiltonian from symbolic expression and build the operator form (sparse matrices or PauliOp, suitable for qiskit quantum circuits).
It considers two types of encoding: 'ed' returns sparse matrix, 'gray' with option sprase=False it returns PauliOp expression, otherwise a sparse matrix.

'Ansaetze.py'
Ansaetze proposal of variational circuit for Gray encoding (for gauge fields) and zero-charge sector (for femrionic d.o.f.).

Old versions:
'Hamiltonian_QED.py'
A python code that builds QED Hamiltonian N-dimensional lattice, both with open and periodic boundary conditions. The formulation considered is from Kogut and Susskind and the Gauss’ law is applied.
By doing so, we will get a gauge invariant system, reduce the number of dynamical links needed
for the computation, and have a more resource-efficient Hamiltonian suitable for a wide range of
quantum hardware.

'Hamiltonianqed.py'
A python code that builds Hamiltonian on a generic N × N lattice, both with open and periodic boundary conditions. The formulation and methods like the other case, only this is limited to 1D and 2D lattices.

![alt text](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/system_2x2_OBC_gausslawTrue.png)
