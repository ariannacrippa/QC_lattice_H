from scipy.sparse.linalg import eigsh
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt

from Hamiltonian_QED_sym import HamiltonianQED_sym
from Hamiltonian_QED_oprt import HamiltonianQED_oprt
from HC_Lattice import HCLattice
#Qiskit import
from qiskit import QuantumCircuit, QuantumRegister,ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter
import qiskit.quantum_info as qinf
from qiskit.primitives import Sampler, Estimator
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.quantum_info import SparsePauliOp, Pauli, Operator
from sys import stdout
from qiskit.algorithms.optimizers import NFT,SLSQP,COBYLA,ADAM,SPSA,QNSPSA,CG,GradientDescent
from qiskit.quantum_info.states import DensityMatrix
import qiskit.quantum_info as qinf

def fidelity_func(myvqd_results2,eigenvectors_matrix,num_eigs):

    """ compute fidelity of num_eigs solutions from vqd algorithm
    Inputs; results from VQD and eigenvector matrix from ED. Update to Qiskit version 0.24.1"""

    vqd_states = [myvqd_results2.optimal_circuits[i].bind_parameters(myvqd_results2.optimal_parameters[i].values()) for i in range(num_eigs)]

    vqd_fid_states =[qinf.Statevector.from_instruction(i) for i in vqd_states]
    ed_fid_states =[Statevector(eigenvectors_matrix[:,i]) for i in range(num_eigs)]

    return [qinf.state_fidelity(vqd,ed) for vqd,ed in zip(vqd_fid_states,ed_fid_states)]


def fidelity_func_e(vqd,ed,index_ed,index_vqd):

    """ compute fidelity of one eigenvector
    Inputs; results from VQD and eigenvector matrix from ED. Update to Qiskit version 0.24.1"""

    vqd_states = vqd.optimal_circuits[index_vqd].bind_parameters(vqd.optimal_parameters[index_vqd].values())

    vqd_fid_states =qinf.Statevector.from_instruction(vqd_states)
    ed_fid_states =Statevector(ed[:,index_ed])

    return qinf.state_fidelity(vqd_fid_states,ed_fid_states)


def upper_bound_fidelity(obs,psi_id,circ_binded,larg_eig,fidelity,eigvec=False):
    """Define an upper nounds for obervables. Based on fidelity.

    |<psi_id|O|psi_id>-<psi|O|psi>| <= 2 sqrt(c) ||O||_infty

    where psu_id is the ideal case, ||O||_infty is the absolute largest
    eigenvalue of the observable (e.g. plaquette exp val is 1)
    and c= 1-|psi_id|psi>|^2 (infidelity).

    If O|psi_id> = eig|psi_id> , i.e. |psi_id> is eigenvector of O
    then

    |<psi_id|O|psi_id>-<psi|O|psi>| <= 2 c ||O||_infty

    See: https://arxiv.org/pdf/2104.00608v2.pdf

    Input params:

    obs:observable must be Pauliops
    psi_id:eigenvector_matrix from exact diagonalisation
    circ_binded: QuantumCircuit ansatz with binded parameters
    larg_eig:largest eigenvalue of obs
    fidelity:fidelity |psi_id|psi>|^2
    eigvec:bool if True  |psi_id> is eigenvector of obs."""

    import scipy.sparse as sp

    c = 1-fidelity

    eig0cj = sp.csr_matrix(psi_id).transpose().conjugate()
    eig0= sp.csr_matrix(psi_id)
    exp_val_id = (eig0*(obs.to_matrix(sparse=True))*eig0cj ).toarray()[0][0].real

    var_eig =qinf.Statevector.from_instruction(circ_binded)
    exp_val_var = var_eig.expectation_value(obs)

    res = np.abs(exp_val_id-exp_val_var)

    if eigvec==False:

        upper = 2*np.sqrt(c)*larg_eig
    else:
        upper = 2*c*larg_eig

    print(res,upper)

    return res

