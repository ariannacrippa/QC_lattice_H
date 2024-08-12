import pytest
from Hamiltonian_QED_oprt import HamiltonianQED_oprt
import qiskit
from HC_Lattice import HCLattice
from Hamiltonian_QED_sym import HamiltonianQED_sym
from qiskit.quantum_info import Statevector
import numpy as np
from itertools import permutations
from sympy import Symbol
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp

test_input_params = {

        'n_sites': [3,2],
        'g': 0.5,
        'fact_e_op': 1,
        'fact_b_op': 1,
        'm': 3,
        'omega': 1,
        'l': 1,
        'L': 8,
        'rotors': False,
        'lambd': 1000,
        'encoding': 'gray',
        'magnectic_basis': False,
        'pbc': False,
        'puregauge': False,
        'static_charges_values': None,
        'display_hamiltoian': False,
        'e_op_out_plus': False,
        'sparse_pauli': True,
        'threshold': 1e-12,
        'num_eigs': 2

    }

def test_matrix_redux(test_input_params):
    
    latt = HCLattice(n_sites,pbc=pbc)

    e_op_free_input_f=[ 'q_01', 'q_11', 'q_21', 'q_20', 'q_10']
    e_op_free_input_g=['E_20y', 'E_10y']
    e_op_free_input=[Symbol(e) for e in e_op_free_input_f[::-1]+e_op_free_input_g[::-1]]

    config = {
            latt: latt,
            'e_op_free_input': e_op_free_input,
            'e_op_out_plus': test_input_params['e_op_out_plus'],
            'n_sites': test_input_params['n_sites'],
            'l': test_input_params['l'],
            'L': test_input_params['L'],
            'encoding': test_input_params['encoding'],
            'magnectic_basis': test_input_params['magnectic_basis'],
            'pbc': test_input_params['pbc'],
            'puregauge': test_input_params['puregauge'],
            'static_charges_values': test_input_params['static_charges_values']
            }
    hamilt_sym = HamiltonianQED_sym(config,display_hamiltonian=False)
    class_H_oprt = HamiltonianQED_oprt( config,hamilt_sym, sparse_pauli=sparse_pauli)

    Hamiltonian_Pauli_tot = class_H_oprt.get_hamiltonian(g_var=test_input_params['g'],m_var=test_input_params['m'],omega=test_input_params['omega'],fact_b_op=test_input_params['fact_b_op'],fact_e_op=test_input_params['fact_e_op'],lambd=test_input_params['lambd'],)
    Hamiltonian_Pauli_tot_cutting = class_H_oprt.get_hamiltonian(g_var=test_input_params['g'],m_var=test_input_params['m'],omega=test_input_params['omega'],fact_b_op=test_input_params['fact_b_op'],fact_e_op=test_input_params['fact_e_op'],lambd=test_input_params['lambd'],cutting=True)

    spectrum, eigenvectors_matrix = eigsh(Hamiltonian_Pauli_tot, k=test_input_params['num_eigs'], which='SA')
    idx = spectrum.argsort()
    spectrum = spectrum[idx]
    eigenvectors_matrix = eigenvectors_matrix[:, idx]


    if class_H_oprt.len_e_op != 0:
        eig0cj = sp.csr_matrix(eigenvectors_matrix[:,0]).transpose().conjugate()
        eig0= sp.csr_matrix(eigenvectors_matrix[:,0])
        plaq = (eig0*h_b_sparse*eig0cj/len(latt.plaq_list) ).toarray()[0][0].real


    spectrum_cutting, eigenvectors_matrix_cutting = eigsh(Hamiltonian_Pauli_tot_cutting, k=test_input_params['num_eigs'], which='SA')
    idx_cutting = spectrum_cutting.argsort()
    spectrum_cutting = spectrum_cutting[idx_cutting]
    eigenvectors_matrix_cutting = eigenvectors_matrix_cutting[:, idx_cutting]

    if class_H_oprt.len_e_op != 0:
        eig0cj_cutting = sp.csr_matrix(eigenvectors_matrix_cutting[:,0]).transpose().conjugate()
        eig0_cutting= sp.csr_matrix(eigenvectors_matrix_cutting[:,0])
        plaq_cutting = (eig0_cutting*h_b_sparse*eig0cj_cutting/len(latt.plaq_list) ).toarray()[0][0].real

    assert np.allclose(spectrum_cutting,spectrum,atol=1e-12)
    assert spectrum_cutting[1] == spectrum[1]



