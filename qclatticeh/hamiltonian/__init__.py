"""
Definition of the Hamiltonian for QED lattice NxN. This includes both symbolic expressions and operator expressions (Pauli op. for qiskit or sparse matrices for ED)
"""

from .Hamiltonian_QED_sym import HamiltonianQED_sym
from .Hamiltonian_QED_oprt import HamiltonianQED_oprt

__all__ = ['HamiltonianQED_sym', 'HamiltonianQED_oprt']
