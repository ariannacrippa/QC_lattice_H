
import numpy as np
import scipy.sparse as sp

def total_charge_op(class_latt, class_H, puregauge,encoding):
    """Operator of total charge, i.e. sum of the charges q00 and q10 on even and odd sites."""
    if puregauge:
        raise ValueError("Empty sites, fermionic part needed")
    if encoding == "gray":
        gauge = class_H.tensor_prod(
            class_H.I, (class_H._n_qubits_g() * (class_H.len_u_op))
        )  # Gray encoding for E fields
    elif encoding == "ed":  # exact diagonaliz. dimensions of gauge fields 2l+1
        gauge = sp.eye((2 * class_H.l_par + 1) ** (class_H.len_u_op),format='csr')

    q10 = -0.5 * (class_H.I + class_H.Z)
    q00 = 0.5 * (class_H.I - class_H.Z)

    #add q00 if f even and q10 if odd
    if class_latt.n_sitestot%2:# odd number of sites (i.e. the last leftmost charge is even, e.g. 3 sites: q00 q10 q00)
        q_tot_op = class_H.pauli_tns(*[q00,]+[class_H.I,]*(class_latt.n_sitestot-1)+[gauge,])
        first=0
    else:#even number of sites (i.e. the last leftmost charge is odd )
        q_tot_op = class_H.pauli_tns(*[q10,]+[class_H.I,]*(class_latt.n_sitestot-1)+[gauge,])
        first=1

    for k in range(1,class_latt.n_sitestot)[::-1]:
        if k%2==first:
            q_tot_op+=class_H.pauli_tns(*[class_H.I,]*(class_latt.n_sitestot-k)+[q00, ]+[class_H.I,]*(k-1)+[gauge,])
        else:
            q_tot_op+=class_H.pauli_tns(*[class_H.I,]*(class_latt.n_sitestot-k)+[q10, ]+[class_H.I,]*(k-1)+[gauge,])

    return q_tot_op


def chiral_condensate(class_latt, class_H, puregauge,encoding):
    """Operator of chiral condensate see https://arxiv.org/pdf/2112.00756.pdf"""
    if puregauge:
        raise ValueError("Empty sites, fermionic part needed")
    if encoding == "gray":
        gauge = class_H.tensor_prod(
            class_H.I, (class_H._n_qubits_g() * (class_H.len_u_op))
        )  # Gray encoding for E fields
    elif encoding == "ed":  # exact diagonaliz. dimensions of gauge fields 2l+1
        gauge = sp.eye((2 * class_H.l_par + 1) ** (class_H.len_u_op),format='csr')

    #n operator for even and odd sites
    n00 = 0.5 * (class_H.I - class_H.Z)
    n10 = 0.5 * (class_H.I + class_H.Z)


    #add n00 if even and n10 if odd
    if class_latt.n_sitestot%2:# odd number of sites (the last site is even n00)
        n_tot_op = class_H.pauli_tns(*[n00,]+[class_H.I,]*(class_latt.n_sitestot-1)+[gauge,])
        first=0
    else:#even number of sites
        n_tot_op = class_H.pauli_tns(*[n10,]+[class_H.I,]*(class_latt.n_sitestot-1)+[gauge,])
        first=1

    for k in range(1,class_latt.n_sitestot)[::-1]:
        if k%2==first:
            n_tot_op+=class_H.pauli_tns(*[class_H.I,]*(class_latt.n_sitestot-k)+[n00, ]+[class_H.I,]*(k-1)+[gauge,])
        else:
            n_tot_op+=class_H.pauli_tns(*[class_H.I,]*(class_latt.n_sitestot-k)+[n10, ]+[class_H.I,]*(k-1)+[gauge,])

    return 1/np.prod(class_latt.n_sites)*n_tot_op