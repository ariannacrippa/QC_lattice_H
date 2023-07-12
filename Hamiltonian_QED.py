# NEW VERSION WITH ONLY HAMILTONIAN
#
"""Definition of the Hamiltonian for QED lattice NxN"""
from __future__ import annotations
import math
import warnings
from functools import reduce
import time
from itertools import permutations, product, combinations
import re
from typing import List
import numpy as np
import networkx as nx
from networkx import all_simple_paths, get_edge_attributes
from networkx.generators.classic import empty_graph
from networkx.utils import pairwise

# from qiskit.opflow import Z, X, Y, I, PauliSumOp, OperatorBase
import qiskit
from qiskit.quantum_info import SparsePauliOp, Pauli, Operator
from IPython.display import display
from scipy import special as sp
import matplotlib.pyplot as plt
from sympy import (
    Symbol,
    symbols,
    solve,
    lambdify,
    Mul,
    Eq,
    latex,
    expand,
    simplify,
    Pow,
    Float,
    Integer,
    cos,
)
from sympy.core.numbers import ImaginaryUnit
from sympy.physics.quantum.dagger import Dagger
from scipy.sparse.linalg import eigs
from scipy import sparse

SPARSE_PAULI = qiskit.quantum_info.operators.symplectic.sparse_pauli_op.SparsePauliOp

# from memory_profiler import profile


# TODO keep sparse pauli op for qiskit circuit(?)
class HamiltonianQED:

    """The algorithm computes the expression of the Quantum Electrodynamics (QED)
    Kogut-Susskind Hamiltonian,
    both in terms of sympy.symbols and in qiskit Pauli matrices for lattices
    from 1D to 3D. The latter formulation is suitable for quantum circuits.
    For fermionic degrees of freedom, the Jordan-Wigner transformation is applied.
    The discretisation of the group U(1) is done by means of group of integer numbers
    Z_(2L+1).In this definition, the Gray encoding is applied for the gauge fields.

    From an instance of a n-dimensional lattice the code generates the Hamiltonian
    related to that lattice.

    The final expression of the Hamiltonian is given in terms of Pauli matrices,
    and it is written by
    following the order right-left, up-down, i.e. the first term is the one acting
    on the rightmost site.

    To define the Hamiltonian, the following parameters are needed in the
    "get_hamiltonian" function:

        g: float or int
        Coupling of the theory.

        m:float or int
            Mass term in the Hamiltonian.

        omega:float or int
            Factor in front of the kinetic Hamiltonian.

        fact_b_op:float or int
            Factor in front of the magnetic Hamiltonian.

        fact_e_op:float or int
            Factor in front of the electric Hamiltonian.

        lambd=int or float
        Parameter for the suppression factor in the Hamiltonian.

    Parameters
    ----------

    lattice: class
        Instance of the class Lattice.

    n_sites: list
        Number of sites in each direction.

    l: int
        Truncation parameter. Defines how many values the gauge fields take,
        e.g. l=1 -> ±1,0 .

    ll: int
        Discretisation parameter L.

    encoding: str
        Encoding used for the gauge fields. Can be "gray" or "ed" (exact diagonalisation).

    magnetic_basis: bool
        If True, then the magnetic basis is considered, False for electric basis.

    pbc : bool
            If `pbc` is True, both dimensions are periodic. If False, none
            are periodic.

    puregauge: bool
        If False, then we have fermionic degrees of freedom in the system, if True only
        gauge fields.

    static_charges_values: dict or None
        If dict, set of static charges with coordinates and values,
        e.g. a charge Q=-1 in site
         (0,0) and a Q=1 in (1,0) will be: {(0,0):-1,(1,0):1}.

    e_op_out_plus:bool
        Boolean variable that specifies if the outgoing electric fields from a certain
        site have positive (True) or negative (False) sign. This definition influences
        the plaquette term and
        the kinetic term (H_k): if + sign then U^dag in H_k / if - then U in H_k.

    display_hamiltonian: bool
        If True, the Hamiltonian and the Gauss law equations are displayed in the output.


    tn_comparison: bool
        If True, it considers only eigenstate within the truncation applied,
        for example if l=1 it will have only configurations that allow for ±1,0 values
        for the gauge fields.

    """

    def __init__(
        self,
        lattice,
        n_sites: list,
        l: int,
        ll: int = 2,
        encoding: str = "gray",
        rotors: bool = False,  # TODO rotors
        magnetic_basis: bool = False,
        pbc: bool = False,
        puregauge: bool = False,
        static_charges_values: dict | None = None,
        e_op_out_plus: bool = False,
        display_hamiltonian: bool = False,
        tn_comparison: bool = False,
    ) -> None:
        self.n_sites = n_sites
        self.pbc = pbc
        self.lattice = lattice
        self.l_par = l
        self.ll_par = ll
        self.encoding = encoding
        self.rotors = rotors
        self.magnetic_basis = magnetic_basis
        self.puregauge = puregauge
        self.static_charges_values = static_charges_values
        self.e_op_out_plus = e_op_out_plus
        self.display_hamiltonian = display_hamiltonian
        self.tn_comparison = tn_comparison

        if self.ll_par <= self.l_par:
            raise ValueError("l must be smaller than L")

        if self.magnetic_basis and self.lattice.dims != 2:
            raise ValueError("Magnetic basis is only implemented for 2D lattices")

        # Pauli matrices
        self.Z = SparsePauliOp(Pauli("Z"))
        self.X = SparsePauliOp(Pauli("X"))
        self.Y = SparsePauliOp(Pauli("Y"))
        self.I = SparsePauliOp(Pauli("I"))

        self._symlist = ["I", "X", "Y", "Z", "Sd", "S-", "Su", "S+"]
        if self.display_hamiltonian:
            print("Alpha angle \u03B1=2 \u03C0/2L+1")
        self.alpha = 2 * np.pi / (2 * self.ll_par + 1) if self.magnetic_basis else 0

        print("HamiltonianQED: Initializing...")
        # get the start time
        start_time = time.time()

        # list of dynamical and static charges
        self.str_node_f = (
            lambda node: str(node)
            if self.lattice.dims == 1
            else "".join(map(str, node))
        )

        self.q_charge_str_list = [
            "q_" + self.str_node_f(node)
            for node in self.lattice.graph.nodes
            if self.puregauge is False
        ]
        self.static_charges_str_list = [
            "Q_" + self.str_node_f(node)
            for node in self.lattice.graph.nodes
            if self.static_charges_values is not None
        ]
        # Dictionary of all elements (E and charges q) in the system with their symbols
        if not self.rotors:
            self.e_op_dict = {
                s_tmp: symbols(s_tmp)
                for s_tmp in self.lattice.list_edges2_e_op
                + self.q_charge_str_list
                + self.static_charges_str_list
            }

            self.u_op_dict = {
                s_tmp: symbols(s_tmp) for s_tmp in self.lattice.list_edges2_u_op
            }
            self.rotor_list = []

        else:
            self.rotor_list = [
                "R_" + self.str_node_f(node) for node in self.lattice.graph.nodes
            ] + [
                "R_" + str(d) for i, d in zip(range(self.lattice.dims), ["x", "y", "z"])
            ]
            self.e_op_dict = {
                s_tmp: symbols(s_tmp)
                for s_tmp in self.rotor_list
                + self.q_charge_str_list
                + self.static_charges_str_list
            }

            self.u_op_dict = {}  # TODO use P

        if not rotors:
            # #Gauss law equations in a list and display them if links not rotors
            self.gauss_equations()
            if self.display_hamiltonian:
                print(">> Gauss law system of equations (symbolic + latex):")
                print(
                    "static charges:",
                    [
                        "Q_" + self.str_node_f(key) + f"={val}"
                        for key, val in self.static_charges_values.items()
                    ] if self.static_charges_values is not None else "None"
                )
                [display(Eq(i, 0)) for i in self.list_gauss]
                [
                    print(latex(i) + " &= 0 \\\\ \\nonumber")
                    for i in self.list_gauss[:-1]
                ]
                print(latex(self.list_gauss[-1]) + " &= 0", "\n")

            # Solution of gauss law equations
            self.sol_gauss = solve(self.list_gauss, dict=True)[0]
            print("> Gauss law equations solved")
            # e_op_free from solution of Guass equations and edges
            self.e_op_free = list(
                set([symbols(j) for j in self.lattice.list_edges2_e_op]).intersection(
                    set(
                        [
                            item
                            for sublist in [
                                eq.free_symbols for eq in self.sol_gauss.values()
                            ]
                            for item in sublist
                        ]
                    )
                )
            )
            # Build u_op_free from e_op_free and edges
            self.u_op_free = [
                k.subs(
                    [
                        (symbols(j), symbols(k))
                        for j, k in zip(
                            self.lattice.list_edges2_e_op, self.lattice.list_edges2_u_op
                        )
                    ]
                )
                for k in self.e_op_free
            ]
            self.u_op_free_dag = [
                k.subs(
                    [
                        (symbols(j), Symbol(k + "D"))
                        for j, k in zip(
                            self.lattice.list_edges2_e_op, self.lattice.list_edges2_u_op
                        )
                    ]
                )
                for k in self.e_op_free
            ]  # U^dag

            # length of e_op_free and u_op_free
            self.len_e_op = len(self.e_op_free)
            self.len_u_op = len(self.u_op_free)
            print("> e_op_free and u_op_free built")
        else:
            self.rotors_conversion()
            print("put rotors here")

        # Define the espressions for substituting symbols into Pauli strings

        # self._symbol_to_pauli()

        self.el_op_enc()
        self.u_op_enc()
        self.u_op_dag_enc()
        self._get_symbol_subs()
        print("> Pauli strings built")
        if display_hamiltonian:
            print(">> Hamiltonian (symbolic + latex):")
        self._hamiltonian_el_autom()
        self._hamiltonian_mag_autom()
        self._hamiltonian_m_autom()
        self._hamiltonian_k_autom()

        self.build_hamiltonian_tot()

        # get the end time
        end_time = time.time()
        # get the execution time
        elapsed_time = end_time - start_time
        print(
            ">> Gauss law applied and Hamiltonian built. ",
            "Execution time:",
            elapsed_time,
            "seconds",
        )

        start_time = time.time()
        self.hamiltonian_suppr()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(">> Suppression term built. ", "Execution time:", elapsed_time, "seconds")

        self.get_hamiltonian()

    def get_hamiltonian(
        self,
        g_var=1.0,
        m_var=1.0,
        omega=1.0,
        fact_b_op=1.0,
        fact_e_op=1.0,
        lambd=1000.0,
    ):
        """Returns the Hamiltonian of the system"""

        # Hamiltonian for fermions
        if self.puregauge:
            self.hamiltonian_ferm = 0
        else:
            self.hamiltonian_ferm = (
                float(omega) * self.hamiltonian_k_pauli
                + float(m_var) * self.hamiltonian_m_pauli
            )
        # Hamiltonian for gauge fields
        self.hamiltonian_gauge = (
            -fact_b_op / (float((g_var) ** 2)) * self.hamiltonian_mag_pauli
            + fact_e_op * float((g_var) ** 2) * self.hamiltonian_el_pauli
        )
        # Final result of the Hamiltonian in terms of Pauli matrices

        hamiltonian_tot = (
            self.hamiltonian_gauge
            + self.hamiltonian_ferm
            + lambd * self.hamiltonian_suppress
        )  # .simplify()

        return hamiltonian_tot

    def _n_qubits_g(self) -> int:
        """Returns the minimum number of qubits required with Gray encoding"""

        return int(np.ceil(np.log2(2 * self.l_par + 1)))

    # Gauss law equations in a list
    def gauss_equations(self):
        """Returns a list of Gauss' law equations (symbols), the system of equation
            can be solved in order to find the set of independent gauge field.

        Returns
        -------

        list_gauss: list of symbols
            List of Gauss' law equations.

        """
        gc_tmp = 0
        list_gauss = []
        for node in self.lattice.graph.nodes:
            if self.puregauge:
                ga_tmp = 0
            else:
                ga_tmp = -1 * self.e_op_dict["q_" + self.str_node_f(node)]
                gc_tmp += self.e_op_dict["q_" + self.str_node_f(node)]

            if self.static_charges_values is not None:
                if node in self.static_charges_values.keys():
                    ga_tmp -= self.static_charges_values[node]

            e_op_i = "E_" + self.str_node_f(node)
            for j, k in zip(self.lattice.list_edges, self.lattice.list_edges2_e_op):
                if e_op_i in j:
                    if e_op_i == j[0]:  # E_out
                        coeff = (
                            1 if self.e_op_out_plus else -1
                        )  # if +1 then U in H_k / if -1 then U^dag in H_k
                    else:  # E_in
                        coeff = (
                            -1 if self.e_op_out_plus else 1
                        )  # if -1 then U in H_k / if 1 then U^dag in H_k

                    ga_tmp += coeff * Symbol(k)

            list_gauss.append(ga_tmp)
        if gc_tmp != 0:
            list_gauss.append(gc_tmp)
        self.list_gauss = list_gauss

    # Tensor product of Pauli matrices
    def tensor_prod(self, pauli, power):
        """Returns tensor product of pauli operator with itself power times"""
        if power == 0:
            return 1
        elif power == 1:
            return pauli
        else:
            return pauli.tensor(self.tensor_prod(pauli, power - 1))

    # multiple tensor product of Pauli matrices
    @staticmethod
    def pauli_tns(*args):
        """Returns Pauli tensor product of all arguments. If int in args then it skips it.
        If all arguments are SparsePauliOp then it applies tensor method of SparsePauliOp.
        If not it applies kronecker product of numpy.(it works also with SparsePauliOp) but much slower.)
        """
        valid_args = [arg for arg in args if not isinstance(arg, int)]

        if len(valid_args) >= 2:
            if all(
                [type(arg) == SPARSE_PAULI for arg in valid_args]
            ):  # all SparsePauliOp
                return reduce(lambda x, y: x.tensor(y), valid_args)
            else:
                return reduce(
                    lambda x, y: sparse.kron(
                        sparse.csr_matrix(x), sparse.csr_matrix(y)
                    ),
                    valid_args,
                )

        elif len(valid_args) == 1:
            return valid_args[0]
        else:
            raise ValueError("Insufficient valid arguments for tensor product")

    # decompose sympy expression into a list of symbols and powers
    @staticmethod
    def decompose_expression(expr):
        """Decompose a sympy expression into a list of symbols and powers."""
        if isinstance(expr, (Symbol, Pow, Float, Integer)):  # single symbol or number
            return [expr]
        elif isinstance(expr, Mul):
            terms = expr.as_ordered_factors()
            result = [t if isinstance(t, (Symbol, Pow)) else t for t in terms]
            return result
        else:
            raise ValueError("Invalid expression type")

    # @profile
    def list_to_enc_hamilt(
        self, list_el, subst, ferm_lst=[], gauge_lst=[], encoding="gray"
    ):
        """Return a list of Pauli operators or list of matrices (depending on the encoding used) from a list of symbolic operators.
        It consider only single operator, not entire pauli string, thus operation like (I^op^I)**2 and I^op1^I*I^op2^I are
        then simplied to I^(op**2)^I and I^(op1*op2)^I, respectively.
        Last part is put everything together and add identity matrices where needed.

        encoding: gray, ed (exact diagonalization)"""
        ham_encoded = np.empty(len(list_el), dtype=object)  # []
        for kk, ei in enumerate(list_el):
            index_op = []  # build index list order ..q2q1q0 (little endian)
            for e in ei:
                if not isinstance(
                    e, (int, float, complex, Float, Integer, str, ImaginaryUnit)
                ):
                    if (
                        list(e.free_symbols)[0].name[-1] == "D"
                        and list(e.free_symbols)[0].name[0] == "U"
                    ):  # gauge field U adjoint
                        index_op.append(
                            str(
                                (
                                    ferm_lst[::-1]
                                    + [Symbol(i.name + "D") for i in gauge_lst][::-1]
                                ).index(*e.free_symbols)
                            )
                            + "D"
                        )
                    elif (
                        list(e.free_symbols)[0].name[-1] == "D"
                        and list(e.free_symbols)[0].name[0:3] == "Phi"
                    ):  # fermion adjoint (but JW index only 0, must cover all the fermionic dof)
                        index_op.append(
                            str(
                                (
                                    [
                                        Symbol(i.name + "D", commutative=False)
                                        for i in ferm_lst
                                    ][::-1]
                                    + gauge_lst[::-1]
                                ).index(*e.free_symbols)
                            )
                            + "D"
                        )
                    else:  # no adjoint
                        index_op.append(
                            str(
                                (ferm_lst[::-1] + gauge_lst[::-1]).index(
                                    *e.free_symbols
                                )
                            )
                        )

            ei_func = lambdify(list(zip(*subst))[0], ei)
            pauli_ei = ei_func(*list(zip(*subst))[1])
            op_dct = {}
            numbers = []
            ct = 0
            for (
                el
            ) in pauli_ei:  # build dictionary of sparse pauli operators and their index
                if isinstance(
                    el,
                    (
                        SparsePauliOp,
                        np.ndarray,
                        sparse._csr.csr_matrix,
                        sparse._coo.coo_matrix,
                    ),
                ):
                    op_dct[index_op[ct]] = el
                    ct += 1
                else:
                    numbers.append(el)
            if subst[0][0] == Symbol("Phi_1D", commutative=False):  # ferm
                res = ["id_f"] * len(ferm_lst) + ["id_g"] * self.len_e_op
                f_index_op = [
                    i for i in index_op if int(re.findall("\d+", i)[0]) < len(ferm_lst)
                ]  # select only fermionic dof
                res[0] = (
                    op_dct[f_index_op[0]] @ op_dct[f_index_op[1]]
                )  # compute product between fermions dof when JW applied
                start = len(ferm_lst)

            else:  # no JW
                res = ["id_q"] * len(ferm_lst) + ["id_g"] * self.len_e_op
                start = 0

            for i in range(start, len(res)):  # only for gauge or charges q
                if str(i) in op_dct.keys() and isinstance(
                    res[i], str
                ):  # fill res with SparsePauli
                    res[i] = op_dct[str(i)]
                if str(i) + "D" in op_dct.keys() and isinstance(
                    res[i], str
                ):  # fill res with SparsePauli
                    res[i] = op_dct[str(i) + "D"]

                if (
                    isinstance(res[i], str) and res[i] == "id_q"
                ):  # remaining spots for charges are filled with I
                    res[i] = self.I.to_matrix(sparse=False)  # single qubit for charge
                elif isinstance(res[i], str) and res[i] == "id_g":
                    if encoding == "gray":
                        res[i] = self.tensor_prod(
                            self.I, (self._n_qubits_g())
                        ).to_matrix(
                            sparse=False
                        )  # Gray encoding for E field
                    elif (
                        encoding == "ed"
                    ):  # exact diagonaliz. dimensions of gauge fields 2l+1
                        res[i] = sparse.eye(2 * self.l_par + 1)
            res = [
                elem for elem in res if not isinstance(elem, str)
            ]  # remove id_f when JW applied

            ham_encoded[kk] = np.prod(numbers) * self.pauli_tns(*res)

        return ham_encoded

    def jw_funct(self, n_tmp: int, n_qubits: int):
        """Jordan-Wigner for 2 terms phi^dag, phi

        Inputs:
            n_tmp: index of fermionic operator

            n_qubits: n.er of total qubits in the string

        """
        # sgm = PauliSumOp( SparsePauliOp.from_sparse_list( [ ( "X", [ 0, ], 0.5, ), ] + [ ( "Y", [ 0, ], (-0.5j), ), ], num_qubits=1, ) )
        # sgp = PauliSumOp( SparsePauliOp.from_sparse_list( [ ( "X", [ 0, ], 0.5, ), ] + [ ( "Y", [ 0, ], (0.5j), ), ], num_qubits=1, ) )
        sgm = SparsePauliOp.from_sparse_list(
            [
                (
                    "X",
                    [
                        0,
                    ],
                    0.5,
                ),
            ]
            + [
                (
                    "Y",
                    [
                        0,
                    ],
                    (-0.5j),
                ),
            ],
            num_qubits=1,
        )
        sgp = SparsePauliOp.from_sparse_list(
            [
                (
                    "X",
                    [
                        0,
                    ],
                    0.5,
                ),
            ]
            + [
                (
                    "Y",
                    [
                        0,
                    ],
                    (0.5j),
                ),
            ],
            num_qubits=1,
        )

        assert n_tmp > 0
        if n_tmp == 1:
            jw_dagk = self.tensor_prod(self.I, 0)  # (I) ^ 0
            jwk = self.tensor_prod(self.I, 0)  # (I) ^ 0

        else:
            jw_dagk = ((1j) ** (n_tmp - 1)) * self.tensor_prod(self.Z, (n_tmp - 1))
            jwk = ((-1j) ** (n_tmp - 1)) * self.tensor_prod(self.Z, (n_tmp - 1))

        jw_dag = HamiltonianQED.pauli_tns(
            self.tensor_prod(self.I, (n_qubits - n_tmp)), (sgm), (jw_dagk)
        )
        jw_nodag = HamiltonianQED.pauli_tns(
            self.tensor_prod(self.I, (n_qubits - n_tmp)), (sgp), (jwk)
        )

        return jw_dag, jw_nodag  # then use: jw_dag@jw_nodag for phi^dag phi

    # utilities and operators
    def _gray_map(self):
        """Gray map dictionary for a certain value of the truncation parameter l.
        for example if l = 1, it returns:{-1: '00', 0: '01', 1: '11'}"""
        ret = {}
        for i in range(0, 2 * self.l_par + 1):
            gray_decimal = i ^ (i >> 1)
            ret[i - self.l_par] = "{0:0{1}b}".format(gray_decimal, self._n_qubits_g())
        return ret

    # Transition map for encoding
    @staticmethod
    def _trans_map(string_1, string_2):
        """Transition map for encoding.
        Parameters:

          string_1: string of the state (output of gray_map)
          string_2: string of the state (output of gray_map)

          Example ouf output with psi_phys = 1
                                  psi_Gray =  '11'
                                  -> output = [6, 6]
        """
        op_dict = {
            ("0", "0"): 4,  # Sd
            ("0", "1"): 5,  # S-
            ("1", "1"): 6,  # Su
            ("1", "0"): 7,  # S+
        }
        n_tmp = len(string_1)
        assert n_tmp == len(string_2)
        symb_list = [op_dict[(a1, a2)] for a1, a2 in zip(string_1, string_2)]
        return symb_list

    # rotator-string/Electric field operator
    def _r_c(self):
        """Rotator-string/Electric field operator.
        Function equivalent to S_z term.
        Defined for Gray encoding"""

        states_list = list(range(-self.l_par, self.l_par + 1))  # states: -l, ..., l
        e_op_list = []
        for st_fact in states_list:
            if st_fact != 0:
                # using the fact that the superposition encoding is only for st_fact=0
                e_op_list.append(
                    [
                        st_fact,
                        *HamiltonianQED._trans_map(
                            self._gray_map()[st_fact], self._gray_map()[st_fact]
                        ),
                    ]
                )
        e_oper = [
            [
                v_elem[0],
                [
                    f"{self._symlist[s_tmp]}_{self._n_qubits_g()-i-1}"
                    for i, s_tmp in enumerate(v_elem[1:])
                ],
            ]
            for v_elem in e_op_list
        ]
        e_oper_str = [
            f"{v_elem[0]} "
            + " ".join(
                f"{self._symlist[s_tmp]}_{self._n_qubits_g()-i-1}"
                for i, s_tmp in enumerate(v_elem[1:])
            )
            for v_elem in e_op_list
        ]
        return e_oper, e_oper_str

    # (downward) ladder operator
    def _l_c(self):
        """Ladder function equivalent to V^- terms
        Defined for Gray encoding"""
        states_list = list(range(-self.l_par, self.l_par + 1))  # states: -l, ..., l

        u_op_list = []
        for st_fact in states_list:
            if st_fact != -self.l_par:
                encs = [self._gray_map()[st_fact - 1], self._gray_map()[st_fact]]
                are_tuples = list(map(lambda x: isinstance(x, tuple), encs))
                nterms = 1 << sum(are_tuples)
                fact = 1.0 / np.sqrt(nterms)
                for ei_tmp, enc in enumerate(encs):
                    if not are_tuples[ei_tmp]:
                        encs[ei_tmp] = (enc,)
                for e1_tmp in encs[0]:
                    for e2_tmp in encs[1]:
                        u_op_list.append(
                            [fact, *HamiltonianQED._trans_map(e1_tmp, e2_tmp)]
                        )
        u_oper = [
            f"{v_elem[0]} "
            + " ".join(
                f"{self._symlist[s_tmp]}_{self._n_qubits_g()-i-1}"
                for i, s_tmp in enumerate(v_elem[1:])
            )
            for v_elem in u_op_list
        ]

        return u_oper

    # @staticmethod
    def str_to_pauli(self, lst: list, n_tmp: int):
        """
        Returns PauliSumOp object.
        Output order of Pauli matrices from right to left: ..q2q1q0

        Inputs

        lst:list of strings of operators in the following opdict
        n_tmp: num_qubits

        Rules:
        Hamiltonian terms made of tensor products of pauli operators as a list of
        (f-)strings.
        Each string contains two parts: 1st encodes the overall coefficient, 2nd one
        encodes the tensor product of operators in the set {I,X,Y,Z,S+,S-,Su,Sd}
        (S± are (X ∓ j Y)/2, while Su=(I-Z)/2 and Sd=(I+Z)/2)
        and the nth-qubit on which they act in terms of a space-separated sequence
        of underscore-separated operator-position pairs, e.g. $Z_2 X_5 S-_3$.
        Note: difference with usual sigma^±. In order to have:
         S+|0>=|1>, S-|1>=|0>  (|0>=(1,0), |1>= (0,1))
        """
        opdict = {
            "S+": [
                ("X", 0.5),
                ("Y", -0.5j),
            ],
            "S-": [("X", 0.5), ("Y", 0.5j)],
            "Su": [("I", 0.5), ("Z", -0.5)],
            "Sd": [("I", 0.5), ("Z", 0.5)],
        }
        pauli_res = 0
        for s_tmp in lst:
            facts = s_tmp.split(" ")
            splitted_facts = list(
                map(lambda x: (int(x[1]), x[0]), map(lambda x: x.split("_"), facts[1:]))
            )

            sparse_prod = self.tensor_prod(self.I, n_tmp)  # I ^ (n_tmp)
            #
            for st_fact in splitted_facts:
                sparse_prod @= SparsePauliOp.from_sparse_list(
                    [
                        (
                            p_tmp[0],
                            [
                                st_fact[0],
                            ],
                            p_tmp[1],
                        )
                        for p_tmp in opdict[st_fact[1]]
                    ],
                    num_qubits=n_tmp,
                )
            # pauli_sum = PauliSumOp(sparse_prod, coeff=complex(facts[0]))

            # pauli_res += pauli_sum
            sparse_prod.coeffs *= complex(facts[0])
            pauli_res += sparse_prod.simplify()

        return pauli_res  # .reduce()

    def el_op_enc(self):
        """Return the encoding of the electric field operator in the chosen encoding"""
        if self.encoding == "gray":
            self.e_oper = self.str_to_pauli(
                self._r_c()[1], self._n_qubits_g()
            ).to_matrix(sparse=True)
        elif self.encoding == "ed":
            self.e_oper = sparse.diags(
                np.arange(-self.l_par, self.l_par + 1), format="csr"
            )  # NB: the operator are all sparse since power matrix M@M=M**2 does not work for non-sparse matrices (i.e. if non-sparse it does power element-wise))
        else:
            raise ValueError("encoding not recognized")

    def u_op_enc(self):
        """Return the encoding of the link operator in the chosen encoding"""
        if self.encoding == "gray":
            self.u_oper = self.str_to_pauli(self._l_c(), self._n_qubits_g()).to_matrix(
                sparse=True
            )
        elif self.encoding == "ed":
            size_op = 2 * self.l_par + 1
            u_ed = np.zeros((size_op, size_op))
            # Fill the upper diagonal with 1s: U
            for i in range(size_op - 1):
                u_ed[i, i + 1] = 1
            self.u_oper = sparse.csr_matrix(
                u_ed
            )  # NB: the operator are all sparse since power matrix M@M=M**2 does not work for non-sparse matrices (i.e. if non-sparse it does power element-wise))
        else:
            raise ValueError("encoding not recognized")

    def u_op_dag_enc(self):
        """Return the encoding of the link operator dagger in the chosen encoding"""
        if self.encoding == "gray":
            self.u_oper_dag = (
                self.str_to_pauli(self._l_c(), self._n_qubits_g())
                .adjoint()
                .to_matrix(sparse=True)
            )
        elif self.encoding == "ed":
            u_ed_dag = np.zeros((2 * self.l_par + 1, 2 * self.l_par + 1))
            # Fill the lower diagonal with 1s: U_dag
            for i in range(2 * self.l_par):
                u_ed_dag[i + 1, i] = 1
            self.u_oper_dag = sparse.csr_matrix(
                u_ed_dag
            )  # NB: the operator are all sparse since power matrix M@M=M**2 does not work for non-sparse matrices (i.e. if non-sparse it does power element-wise))
        else:
            raise ValueError("encoding not recognized")

    @staticmethod
    def hermitian_c(expr):
        """Compute hermitian conjugate of input expr."""

        if isinstance(expr, np.ndarray):
            return np.conj(expr).T
        elif isinstance(expr, (sparse._csr.csr_matrix, sparse._coo.coo_matrix)):
            return expr.transpose().conjugate()
        elif isinstance(expr, SparsePauliOp):
            return expr.adjoint()
        else:
            raise ValueError("encoding not recognized for hermitian conjugate")

    def matx_exp(self, matrix, coefficient):
        """Compute the matrix exponential of a matrix using the eigendecomposition
        Input arguments:operator SparsePauliOp
        coefficient = coefficient of the exponential"""
        if isinstance(matrix, (sparse._csr.csr_matrix, sparse._coo.coo_matrix)):
            matrix = matrix.toarray()
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        diagonal_matrix = np.diag(np.exp(coefficient * eigenvalues))
        exp_matrix = eigenvectors @ diagonal_matrix @ np.linalg.inv(eigenvectors)
        if isinstance(
            matrix, (np.ndarray, sparse._csr.csr_matrix, sparse._coo.coo_matrix)
        ):
            return sparse.csr_matrix(exp_matrix)
        elif isinstance(matrix, SparsePauliOp):
            return SparsePauliOp.from_operator(
                Operator(exp_matrix)
            )  # NB attention this operation is exponential

    def cos_oper(self, exp_matrixp):
        """Compute the cosine of a matrix using Euler formula,
        cos(operators)=exp(i*operators)+exp(-i*operators)/2

        """
        if isinstance(exp_matrixp, np.ndarray):
            return (exp_matrixp + exp_matrixp.conj().T) / 2
        elif isinstance(exp_matrixp, (sparse._csr.csr_matrix, sparse._coo.coo_matrix)):
            return (exp_matrixp + exp_matrixp.conjugate().transpose()) / 2
        elif isinstance(exp_matrixp, SparsePauliOp):
            return (exp_matrixp + exp_matrixp.adjoint()) / 2
        else:
            raise ValueError("encoding not recognized")

    def sin_oper(self, exp_matrixp):
        """Compute the sine of a matrix using Euler formula,
        sin(operators)=exp(i*operators)-exp(-i*operators)/2

        """
        if isinstance(exp_matrixp, np.ndarray):
            return (exp_matrixp - exp_matrixp.conj().T) / 2j
        elif isinstance(exp_matrixp, (sparse._csr.csr_matrix, sparse._coo.coo_matrix)):
            return (exp_matrixp - exp_matrixp.conjugate().transpose()) / 2j
        elif isinstance(exp_matrixp, SparsePauliOp):
            return (exp_matrixp - exp_matrixp.adjoint()) / 2j
        else:
            raise ValueError("encoding not recognized")

    def rotors_conversion(self):  # TODO test this function
        rotors_dict = {}
        for (
            s
        ) in (
            self.lattice.list_edges2_e_op
        ):  # index of rotors is the bottom left index (nx,ny coordinates)
            coord_s = re.findall(r"\d+", s)[0]
            if coord_s[0] == "0" and coord_s[1] == str(
                self.n_sites[1]
            ):  # if nx==0 and ny!=n_y_max
                eop_tmp = 0
            else:
                eop_tmp = Symbol("R_" + coord_s)

            if s[-1] == "x":
                if coord_s[1] == "0":  # ny==0
                    eop_tmp += Symbol("R_x")
                    if self.pbc:  # only if pbc in y direction
                        eop_tmp -= Symbol("R_" + coord_s[0] + str(self.n_sites[1] - 1))

                else:
                    eop_tmp -= Symbol("R_" + coord_s[0] + str(int(coord_s[1]) - 1))

                if not self.puregauge:
                    q_tmp = -sum(
                        [
                            Symbol("q_" + str(x) + str(y))
                            for x in range(int(coord_s[0]) + 1, self.n_sites[0])
                            for y in range(self.n_sites[1])
                            if int(coord_s[1]) == 0
                        ]
                    )
                    eop_tmp += q_tmp

            elif s[-1] == "y":
                eop_tmp *= -1
                if coord_s[0] == "0":  # if n_x ==0
                    eop_tmp += Symbol("R_y")
                    if self.pbc:  # only if pbc in x direction
                        eop_tmp += Symbol("R_" + str(self.n_sites[0] - 1) + coord_s[1])
                else:
                    eop_tmp += Symbol("R_" + str(int(coord_s[0]) - 1) + coord_s[1])

                if not self.puregauge:
                    q_tmp = -sum(
                        [
                            Symbol("q_" + str(x) + str(y))
                            for x in range(self.n_sites[0])
                            for y in range(int(coord_s[1]) + 1, self.n_sites[1])
                            if x == int(coord_s[0])
                        ]
                    )
                    eop_tmp += q_tmp

            # if coordinates of R_xy are 0 and n_sites-1, then R is 0 convenient to fix this to zero
            for i in eop_tmp.free_symbols:
                if i.name[0] == "R" and i.name[2] == "0":
                    if i.name[3] == str(self.n_sites[1] - 1):
                        eop_tmp = eop_tmp.subs(i, 0)

            rotors_dict[s] = eop_tmp

        self.rotors_dict = rotors_dict

    def _get_symbol_subs(self):
        """Return list of substitutions for symbols in the Hamiltonian.
        Suitable for every encoding (gauge fields) defined in el_op_enc(), u_op_enc(), u_op_dag_enc()
        """
        # list of symbols only (encoding not needed)
        self.eop_list = self.e_op_free if not self.rotors else self.rotor_list
        self.uop_list = self.u_op_free if not self.rotors else []
        self.qop_list = (
            []
            if self.puregauge
            else [symbols("q_" + self.str_node_f(k)) for k in self.lattice.jw_sites]
        )
        self.static_qop_list = (
            []
            if self.static_charges_values is None
            else [symbols("Q_" + self.str_node_f(k)) for k in self.lattice.jw_sites]
        )
        # self.qop_list += self.static_qop_list

        self.phiop_list = [
            Symbol(f"Phi_{i+1}", commutative=False)
            for i, k in enumerate(self.lattice.jw_sites)
        ]

        q10 = -0.5 * (self.I + self.Z)  # JW dependent
        q00 = 0.5 * (self.I - self.Z)

        sum_k = lambda k: k if self.lattice.dims == 1 else sum(k)

        # list of symbols and operators
        self.qcharge_list = [
            (symbols("q_" + self.str_node_f(k)), (q10 if sum_k(k) % 2 else q00))
            for k in self.lattice.jw_sites
        ]
        if self.static_charges_values is None:
            self.static_qcharge_list = []
        else:
            self.static_qcharge_list = [
                (
                    Symbol("Q_" + self.str_node_f(node)),
                    self.static_charges_values[node] * self.I,
                )
                if node in self.static_charges_values.keys()
                else (Symbol("Q_" + self.str_node_f(node)), 0 * self.I)
                for node in self.lattice.graph.nodes
            ]
        # self.qcharge_list += self.static_qcharge_list

        self.e_field_list = [(s_tmp, self.e_oper) for s_tmp in self.eop_list]
        self.u_field_list = [
            (s_tmp, (self.u_oper_dag if s_tmp.name[-1] == "D" else self.u_oper))
            for s_tmp in self.uop_list + self.u_op_free_dag
        ]
        ei_class = lambda fct: self.matx_exp(
            fct * self.e_oper, 1j * self.alpha
        )  # compute exp(i*alpha*E)
        # U for mag basis (used in H_kinetic_mag)
        self.u_field_list_mag = [
            (s_tmp, (ei_class(1) if s_tmp.name[-1] == "D" else ei_class(-1)))
            for s_tmp in self.uop_list + self.u_op_free_dag
        ]  # U->exp(-i*alpha*E), U_dag->exp(i*alpha*E) in mag basis

        phi_el = lambda i, j: HamiltonianQED.pauli_tns(
            (self.jw_funct(i + 1, int(self.lattice.n_sitestot))[j]),
        )
        self.phi_jw_list = [
            (
                Symbol(f"Phi_{i+1}D", commutative=False),
                phi_el(i, 0),
            )
            for i, k in enumerate(self.lattice.jw_sites)
        ] + [
            (
                Symbol(f"Phi_{i+1}", commutative=False),
                phi_el(i, 1),
            )
            for i, k in enumerate(self.lattice.jw_sites)
        ]

    # HAMILTONIAN
    # * symbols
    # Define Hamiltonian and apply Gauss laws

    def _hamiltonian_el_autom(self):
        """Hamiltonian for E field"""
        hamiltonian_el_sym = [Symbol(str(s)) for s in self.lattice.list_edges2_e_op]
        if not self.rotors:
            hamiltonian_el_sym = sum(
                [
                    x**2 if x not in self.sol_gauss else (self.sol_gauss[x]) ** 2
                    for x in hamiltonian_el_sym
                ]
            )  # Gauss law applied
        else:  # if rotors not considered gauss law , they already satisfy it
            hamiltonian_el_sym = sum(
                [(x.subs(self.rotors_dict)) ** 2 for x in hamiltonian_el_sym]
            )

        self.hamiltonian_el_sym = (
            hamiltonian_el_sym  # symbolic expression (useful for diplay)
        )

        if self.magnetic_basis:
            fnu_sin = lambda nu: float(
                (-1) ** (nu + 1)
                / (2 * np.pi)
                * (
                    sp.polygamma(
                        0, (2 * self.ll_par + 1 + nu) / (2 * (2 * self.ll_par + 1))
                    )
                    - sp.polygamma(0, nu / (2 * (2 * self.ll_par + 1)))
                )
            )  # f_nu^s factor for E operator

            fnu_cos = lambda nu: float(
                (-1) ** (nu)
                / (4 * np.pi**2)
                * (
                    sp.polygamma(1, nu / (2 * (2 * self.ll_par + 1)))
                    - sp.polygamma(
                        1, (2 * self.ll_par + 1 + nu) / (2 * (2 * self.ll_par + 1))
                    )
                )
            )  # f_nu^c factor for E^e operator

            # dict for substitution of E operators to expression of U and U^dag for magnetic basis
            E_mag_subs = lambda nu: {
                el_eop: fnu_sin(nu)
                * (
                    Symbol("U_" + el_eop.name[2:]) ** nu
                    - Symbol("U_" + el_eop.name[2:] + "D") ** nu
                )
                / (2j)
                for el_eop in self.e_op_free
            }
            # dict for substitution of E^2 operators to expression of U and U^dag for magnetic basis
            Epow2_mag_subs = lambda nu: {
                el_eop**2: fnu_cos(nu)
                * (
                    Symbol("U_" + el_eop.name[2:]) ** nu
                    + Symbol("U_" + el_eop.name[2:] + "D") ** nu
                )
                / 2
                + Symbol("L")
                for el_eop in self.e_op_free
            }

            hamilt_el_expand = expand(self.hamiltonian_el_sym)
            hamiltonian_el_sym_mbasis = []
            for nu in range(1, 2 * self.ll_par + 1):  # for loop over nu
                ham_subs = hamilt_el_expand.subs(Epow2_mag_subs(nu)).subs(
                    E_mag_subs(nu)
                )
                if nu > 1:  # the factor with L is independent of sum over nu
                    ham_subs = ham_subs.subs(Symbol("L"), 0)
                hamiltonian_el_sym_mbasis.append(
                    ham_subs.subs(Symbol("L"), self.ll_par * (self.ll_par + 1) / 3)
                )
            hamiltonian_el_sym_mbasis = sum(hamiltonian_el_sym_mbasis)

            self.hamiltonian_el_sym_mbasis = (
                hamiltonian_el_sym_mbasis  # symbolic expression (useful for diplay)
            )

            self.hamiltonian_el_subs = list(
                hamiltonian_el_sym_mbasis.expand().args
            )  # list of symbolic expressions (must use expand for now. otherwise error in pauli substitution)
            print("Magnetic basis used for electric H")
        else:
            self.hamiltonian_el_subs = list(
                hamiltonian_el_sym.expand().args
            )  # list of symbolic expressions

    def _hamiltonian_mag_autom(self):
        """Hamiltonian for B field"""
        plaq_u_op_gaus = [
            [
                x if symbols(x) in self.u_op_free else "iD"
                for x in [k for j, k in enumerate(p_tmp)]
            ]
            for p_tmp in self.lattice.list_plaq_u_op
        ]

        # Hamiltonian for substitution
        hamiltonian_mag_subs = [
            [
                symbols(k).subs(symbols("iD"), 1)
                if j < 2
                else Symbol(k + "D").subs(symbols("iDD"), 1)
                for j, k in enumerate(p_tmp)
            ]
            for p_tmp in plaq_u_op_gaus
        ]

        self.plaq_u_op_gaus = plaq_u_op_gaus
        self.hamiltonian_mag_subs = hamiltonian_mag_subs

    def _hamiltonian_m_autom(self):
        """Hamiltonian for mass term
        Returns a list of symbols for mass Hamiltonian of the type 'phi^dag phi'.

        Returns
        -------

        hamiltonian_m_sym:list
        List of tuples like [(Phi_1D, Phi_1), (Phi_2D, Phi_2),..]

        """
        hamiltonian_m_sym = []
        # dictionary for fermionic sistes to symbols

        jw_dict = {
            k: (
                Symbol(f"Phi_{i+1}D", commutative=False),
                Symbol(f"Phi_{i+1}", commutative=False),
            )
            for i, k in enumerate(self.lattice.jw_sites)
        }

        for i in jw_dict:
            hamiltonian_m_sym.append((jw_dict[i][0], jw_dict[i][1]))

        self.hamiltonian_m_sym = hamiltonian_m_sym

    def _hamiltonian_k_autom(self):
        """Hamiltonian for kinetic term of the type 'phi^dag U phi'."""

        # dictionary for dynamical links to symbols
        lu_op_edges = [
            [Symbol(k) for k in self.lattice.list_edges2_u_op].index(n_tmp)
            for n_tmp in self.u_op_free
        ]
        u_op_free_edges = [
            (
                tuple(map(int, re.findall(r"\d+", self.lattice.list_edges[i][0])[0])),
                tuple(map(int, re.findall(r"\d+", self.lattice.list_edges[i][1])[0])),
                u_elem,
                udag,
            )
            for i, u_elem, udag in zip(lu_op_edges, self.u_op_free, self.u_op_free_dag)
        ]
        u_op_free_dict = {(k[0], k[1]): (k[2], k[3]) for k in u_op_free_edges}

        # dictionary for fermionic sistes to symbols
        jw_dict = {
            k: (
                Symbol(f"Phi_{i+1}D", commutative=False),
                Symbol(f"Phi_{i+1}", commutative=False),
            )
            for i, k in enumerate(self.lattice.jw_sites)
        }

        # Build Hamiltonian
        hamiltonian_k_sym = []
        for i in self.lattice.graph_edges_system:  # for every edge
            if i in u_op_free_dict:  # if dynamical link
                hamilt_k_elem = (
                    u_op_free_dict[i][1]
                    if self.e_op_out_plus
                    else u_op_free_dict[i][
                        0
                    ]  # u_op_free_dict[i][0] if e_op_out_plus else u_op_free_dict[i][1]
                )  # if Gauss law with E out + -> U / else U^dag
            else:
                hamilt_k_elem = 1
            # phase in H_k in y-direction as Kogut Susskind H #TODO:assume 2 components spinor >check with 4 components

            if self.lattice.dims == 1:
                phase = 1
                hamiltonian_k_sym.append(
                    (phase, jw_dict[i[0]][0], hamilt_k_elem, jw_dict[i[1]][1])
                )

            elif self.lattice.dims == 2:
                phase = (
                    (-1) ** (sum(i[0]) % 2) if i[0][1] != i[1][1] else 1
                )  # change in y direction if x is odd
                xy_term = (
                    "y" if i[0][1] != i[1][1] else "x"
                )  # if x - adjoint, if y + adjoint

                hamiltonian_k_sym.append(
                    (xy_term, phase, jw_dict[i[0]][0], hamilt_k_elem, jw_dict[i[1]][1])
                )

            elif self.lattice.dims == 3:
                # x-direction
                if i[0][0] != i[1][0]:
                    phase = 1
                # y-direction
                elif i[0][1] != i[1][1]:
                    phase = (-1) ** ((sum(i[0][:2]) + 1) % 2)
                # z-direction
                elif i[0][2] != i[1][2]:
                    phase = (-1) ** (sum(i[0][:2]) % 2)

                i_term = (
                    "x"
                    if i[0][0] != i[1][0]
                    else "y"
                    if i[0][1] != i[1][1]
                    else "z"
                    if i[0][2] != i[1][2]
                    else None
                )

                hamiltonian_k_sym.append(
                    (i_term, phase, jw_dict[i[0]][0], hamilt_k_elem, jw_dict[i[1]][1])
                )  # phi^dag U phi

            else:
                raise ValueError("Only 1, 2 and 3 dimensions are supported.")

        self.hamiltonian_k_sym = hamiltonian_k_sym

    # build H

    def build_hamiltonian_tot(self):
        """Builds the total Hamiltonian of the system."""
        # ************************************  H_E   ************************************
        if self.len_e_op > 0:
            if self.magnetic_basis:
                # Pauli expression, since mag basis H_E is in terms of U and U^dag we use u_op_field_subs
                hamiltonian_el_pauli = self.list_to_enc_hamilt(
                    [self.decompose_expression(i) for i in self.hamiltonian_el_subs],
                    self.qcharge_list + self.u_field_list,
                    self.qop_list,
                    self.uop_list,
                    encoding=self.encoding,
                )
            else:
                hamiltonian_el_pauli = self.list_to_enc_hamilt(
                    [self.decompose_expression(i) for i in self.hamiltonian_el_subs],
                    self.qcharge_list + self.e_field_list,
                    self.qop_list,
                    self.eop_list,
                    encoding=self.encoding,
                )

            hamiltonian_el_pauli = (
                np.sum(hamiltonian_el_pauli) / 2
            )  # (must be then multiplied by g^2)

            if self.display_hamiltonian:  # Hamiltonian to print
                h_el_embasis = (
                    self.hamiltonian_el_sym_mbasis
                    if self.magnetic_basis
                    else self.hamiltonian_el_sym
                )
                display_hamiltonian_el = Eq(
                    Symbol("H_E"), (Symbol("g") ** 2) / 2 * h_el_embasis
                )
                display(display_hamiltonian_el)
                print(latex(display_hamiltonian_el))
        else:  # no gauge fields (e.g. 1d OBC case)
            hamiltonian_el_pauli = 0.0 * self.tensor_prod(
                self.I,
                (int(self.lattice.n_sitestot) + self._n_qubits_g() * (self.len_u_op)),
            )

        # ************************************  H_B   ************************************
        if len(self.u_op_free) > 0:  # and self.lattice.dims > 1:
            # Pauli expression

            if self.magnetic_basis:
                U_mag_subs = {
                    **{
                        el_uop: Symbol("E_" + el_uop.name[2:])
                        for el_uop in self.u_op_free
                    },
                    **{
                        el_uop: Symbol("E_" + el_uop.name[2:-1] + "m")
                        for el_uop in self.u_op_free_dag
                    },
                }
                hamiltonian_mag_sym = [
                    [U_mag_subs.get(item, item) for item in sublst if item != 1]
                    for sublst in self.hamiltonian_mag_subs
                ]
                e_op_dict_mbasis = dict(self.e_field_list)
                # compute exp(i*alpha*E)
                ei_class = lambda fct: self.matx_exp(fct * self.e_oper, 1j * self.alpha)

                hamiltonian_mag_pauli = []
                for ei in hamiltonian_mag_sym:
                    id_eop = {}
                    for e in ei:
                        if e in e_op_dict_mbasis.keys():
                            id_eop[list(e_op_dict_mbasis.keys()).index(e)] = 1  #'p'== +
                        else:
                            id_eop[
                                list(e_op_dict_mbasis.keys()).index(Symbol(e.name[:-1]))
                            ] = -1  #'m'== -

                    if self.encoding == "gray":
                        idx = self.tensor_prod(
                            self.I, (self._n_qubits_g())
                        )  # Gray encoding for E field
                    elif (
                        self.encoding == "ed"
                    ):  # exact diagonaliz. dimensions of gauge fields 2l+1
                        idx = sparse.eye(2 * self.l_par + 1, format="csr")

                    if len(ei) == 1:  # cos 1 operator is enough and rest is I
                        cos1 = [
                            self.cos_oper(ei_class(id_eop[i])) if i in id_eop else idx
                            for i in range(self.len_e_op)[::-1]
                        ]  # inverse because little endian
                        hamiltonian_mag_pauli.append(self.pauli_tns(*cos1))

                    else:
                        # compute cosine of multiple operators cos(E1+E2+...)=e^iE1 e^iE2 ... + e^-iE1 e^-iE2 ... /2
                        cosn = self.cos_oper(
                            self.pauli_tns(
                                *[
                                    ei_class(id_eop[i]) if i in id_eop else idx
                                    for i in range(self.len_e_op)
                                ][min(id_eop) : max(id_eop) + 1][::-1]
                            )
                        )
                        # combine with identities, e.g. cos(E2+E3) -> I^cos(E2+E3)^I^I (q4^q3^q2^q1^q0)
                        hamiltonian_mag_pauli.append(
                            self.pauli_tns(
                                *[idx for i in range(self.len_e_op)[::-1]][
                                    max(id_eop) + 1 :
                                ]
                                + [cosn]
                                + [idx for i in range(self.len_e_op)[::-1]][
                                    : min(id_eop)
                                ]
                            )
                        )

                hamiltonian_mag_pauli = (
                    np.sum(hamiltonian_mag_pauli)
                    if self.puregauge
                    else self.pauli_tns(
                        self.tensor_prod(self.I, (int(self.lattice.n_sitestot))),
                        np.sum(hamiltonian_mag_pauli),
                    )
                )  # (must be then multiplied by -1/g^2)
                print("Hamiltonian B mag basis: done")
                if self.display_hamiltonian:
                    # Hamiltonian to print
                    display_hamiltonian_mag = Eq(
                        Symbol("H_B"),
                        -1
                        / (Symbol("g") ** 2)
                        * (
                            sum(
                                [
                                    cos(
                                        Symbol("\u03B1")
                                        * sum(
                                            [
                                                Symbol("E" + k[1:])
                                                if j < 2
                                                else -Symbol("E" + k[1:])
                                                for j, k in enumerate(p_tmp)
                                            ]
                                        ).subs(Symbol("ED"), 0)
                                    )
                                    for p_tmp in self.plaq_u_op_gaus
                                ]
                            )
                        ),
                    )
                    display(display_hamiltonian_mag)
                    print(latex(display_hamiltonian_mag))
            else:
                hamiltonian_mag_pauli = self.list_to_enc_hamilt(
                    self.hamiltonian_mag_subs,
                    self.u_field_list,
                    self.qop_list,
                    self.uop_list,
                    encoding=self.encoding,
                )
                hamiltonian_mag_pauli = (
                    np.sum(hamiltonian_mag_pauli)
                    + self.hermitian_c(np.sum(hamiltonian_mag_pauli))
                ) / 2  # (must be then multiplied by -1/g^2)

                if self.display_hamiltonian:
                    # Hamiltonian to print
                    display_hamiltonian_mag = Eq(
                        Symbol("H_B"),
                        -1
                        / (2 * Symbol("g") ** 2)
                        * (
                            sum(
                                [
                                    np.prod(
                                        [
                                            Symbol(k, commutative=False)
                                            if j < 2
                                            else Dagger(Symbol(k, commutative=False))
                                            for j, k in enumerate(p_tmp)
                                        ]
                                    ).subs(Symbol("iD", commutative=False), 1)
                                    for p_tmp in self.plaq_u_op_gaus
                                ]
                            )
                            + Symbol("h.c.", commutative=False)
                        ),
                    )
                    display(display_hamiltonian_mag)
                    print(latex(display_hamiltonian_mag))
        else:  # no gauge fields (e.g. 1d OBC case)
            hamiltonian_mag_pauli = 0.0 * self.tensor_prod(
                self.I,
                (int(self.lattice.n_sitestot) + self._n_qubits_g() * (self.len_u_op)),
            )
        if not self.puregauge:
            # ************************************  H_K   ************************************
            # Pauli expression of the kinetic term
            if self.magnetic_basis:
                subst_list_hk = self.phi_jw_list + self.u_field_list_mag
            else:
                subst_list_hk = self.phi_jw_list + self.u_field_list

            if self.lattice.dims == 1:
                hamiltonian_k_1x = np.sum(
                    self.list_to_enc_hamilt(
                        [h for h in self.hamiltonian_k_sym],
                        subst_list_hk,
                        self.phiop_list,
                        self.uop_list,
                        encoding=self.encoding,
                    )
                )

                hamiltonian_k_pauli = (
                    0.5j * (hamiltonian_k_1x - self.hermitian_c(hamiltonian_k_1x))
                ).simplify()  # (must be then multiplied by omega)

            elif self.lattice.dims == 2:
                hamiltonian_k_1y = np.sum(
                    self.list_to_enc_hamilt(
                        [h[1:] for h in self.hamiltonian_k_sym if h[0] == "y"],
                        subst_list_hk,
                        self.phiop_list,
                        self.uop_list,
                        encoding=self.encoding,
                    )
                )
                hamiltonian_k_1x = np.sum(
                    self.list_to_enc_hamilt(
                        [h[1:] for h in self.hamiltonian_k_sym if h[0] == "x"],
                        subst_list_hk,
                        self.phiop_list,
                        self.uop_list,
                        encoding=self.encoding,
                    )
                )

                hamiltonian_k_pauli = 0.5j * (
                    hamiltonian_k_1x - self.hermitian_c(hamiltonian_k_1x)
                ) - 0.5 * (
                    hamiltonian_k_1y + self.hermitian_c(hamiltonian_k_1y)
                )  # (must be then multiplied by omega)

            elif self.lattice.dims == 3:
                hamiltonian_k_1y = np.sum(
                    self.list_to_enc_hamilt(
                        [h[1:] for h in self.hamiltonian_k_sym if h[0] == "y"],
                        subst_list_hk,
                        self.phiop_list,
                        self.uop_list,
                        encoding=self.encoding,
                    )
                )
                hamiltonian_k_1x = np.sum(
                    self.list_to_enc_hamilt(
                        [h[1:] for h in self.hamiltonian_k_sym if h[0] == "x"],
                        subst_list_hk,
                        self.phiop_list,
                        self.uop_list,
                        encoding=self.encoding,
                    )
                )
                hamiltonian_k_1z = np.sum(
                    self.list_to_enc_hamilt(
                        [h[1:] for h in self.hamiltonian_k_sym if h[0] == "z"],
                        subst_list_hk,
                        self.phiop_list,
                        self.uop_list,
                        encoding=self.encoding,
                    )
                )
                hamiltonian_k_pauli = (
                    0.5j * (hamiltonian_k_1x - self.hermitian_c(hamiltonian_k_1x))
                    - 0.5 * (hamiltonian_k_1y + self.hermitian_c(hamiltonian_k_1y))
                    + 0.5j * (hamiltonian_k_1z - self.hermitian_c(hamiltonian_k_1z))
                )  # (must be then multiplied by omega)

            else:
                raise ValueError("Dimension not supported")

            if self.display_hamiltonian:
                # Hamiltonian to print

                if self.lattice.dims == 1:
                    if self.magnetic_basis:
                        gauge_fdag = (
                            lambda k: Symbol(
                                "e^{i \u03B1 E_{" + str(k[2])[2:-1] + "}}",
                                commutative=False,
                            )
                            if not isinstance(k[2], int)
                            else k[2]
                        )
                        gauge_f = (
                            lambda k: Symbol(
                                "e^{-i \u03B1 E_{" + str(k[2])[2:-1] + "}}",
                                commutative=False,
                            )
                            if not isinstance(k[2], int)
                            else k[2]
                        )
                    else:
                        gauge_fdag = lambda k: Dagger(
                            Symbol(str(k[2])[:-1], commutative=False)
                        )
                        gauge_f = lambda k: k[2]
                    hamiltonian_k_display = [
                        (
                            k[0],
                            Dagger(Symbol(str(k[1])[:-1], commutative=False)),
                            gauge_fdag(k),
                            k[3],
                        )
                        if str(k[2])[-1] == "d"
                        else (
                            k[0],
                            Dagger(Symbol(str(k[1])[:-1], commutative=False)),
                            gauge_f(k),
                            k[3],
                        )
                        for k in self.hamiltonian_k_sym
                    ]

                    display_hamiltonian_k = Eq(
                        Symbol("H_K"),
                        (Symbol("Omega") * 1j / 2)
                        * (
                            sum(
                                [
                                    Mul(*k, evaluate=False) if k[2] != 1 else Mul(*k)
                                    for k in hamiltonian_k_display
                                ]
                            )
                            - Symbol("h.c.", commutative=False)
                        ),
                        evaluate=False,
                    )

                else:
                    if self.magnetic_basis:
                        gauge_fdag = (
                            lambda k: Symbol(
                                "e^{i \u03B1 E_{" + str(k[3])[2:-1] + "}}",
                                commutative=False,
                            )
                            if not isinstance(k[3], int)
                            else k[3]
                        )
                        gauge_f = (
                            lambda k: Symbol(
                                "e^{-i \u03B1 E_{" + str(k[3])[2:-1] + "}}",
                                commutative=False,
                            )
                            if not isinstance(k[3], int)
                            else k[3]
                        )
                    else:
                        gauge_fdag = lambda k: Dagger(
                            Symbol(str(k[3])[:-1], commutative=False)
                        )
                        gauge_f = lambda k: k[3]

                    hamiltonian_k_display = [
                        (
                            k[1],
                            Dagger(Symbol(str(k[2])[:-1], commutative=False)),
                            gauge_fdag(k),
                            k[4],
                        )
                        if str(k[3])[-1] == "D"
                        else (
                            k[1],
                            Dagger(Symbol(str(k[2])[:-1], commutative=False)),
                            gauge_f(k),
                            k[4],
                        )
                        for k in self.hamiltonian_k_sym
                    ]
                    h_k_x_disp = 0
                    h_k_y_disp = 0
                    h_k_z_disp = 0
                    for k, j in zip(hamiltonian_k_display, self.hamiltonian_k_sym):
                        if j[0] == "x":
                            h_k_x_disp += sum(
                                [Mul(*k, evaluate=False) if k[2] != 1 else Mul(*k)]
                            )
                        elif j[0] == "y":
                            h_k_y_disp += sum(
                                [Mul(*k, evaluate=False) if k[2] != 1 else Mul(*k)]
                            )
                        elif j[0] == "z":
                            h_k_z_disp += sum(
                                [Mul(*k, evaluate=False) if k[2] != 1 else Mul(*k)]
                            )

                    if self.lattice.dims == 3:
                        h_k_z = 0.5j * (
                            h_k_z_disp - Symbol("h.c.(z)", commutative=False)
                        )
                    else:
                        h_k_z = 0
                    display_hamiltonian_k = Eq(
                        Symbol("H_K"),
                        (Symbol("Omega"))
                        * (
                            0.5j * (h_k_x_disp - Symbol("h.c.(x)", commutative=False))
                            - 0.5 * (h_k_y_disp + Symbol("h.c.(y)", commutative=False))
                            + h_k_z
                        ),
                        evaluate=False,
                    )

                display(display_hamiltonian_k)
                print(latex(display_hamiltonian_k))
            # ************************************  H_M   ************************************
            # H_M in terms of Paulis
            # hamiltonian_m_pauli = sum(
            #     [
            #         (-1) ** j
            #         * self.subs_hamilt_sym_to_pauli(h, self.phi_jw_subs)
            #         for j, h in enumerate(self.hamiltonian_m_sym)
            #     ]
            # )  # (must be then multiplied by m)
            hamiltonian_m_pauli = np.sum(
                [
                    ((-1) ** j) * el
                    for j, el in enumerate(
                        self.list_to_enc_hamilt(
                            self.hamiltonian_m_sym,
                            self.phi_jw_list,
                            self.phiop_list,
                            encoding=self.encoding,
                        )
                    )
                ]
            )  # (must be then multiplied by m)
            if self.display_hamiltonian:  # to print
                display_hamiltonian_m = Eq(
                    Symbol("H_m"),
                    Symbol("m")
                    * sum(
                        [
                            (-1) ** j * np.prod(k)
                            for j, k in enumerate(
                                [
                                    (k[0].subs(k[0], Dagger(k[1])), k[1])
                                    for k in self.hamiltonian_m_sym
                                ]
                            )
                        ]
                    ),
                )

                display(display_hamiltonian_m)
                print(latex(display_hamiltonian_m))

            self.hamiltonian_k_pauli = hamiltonian_k_pauli
            self.hamiltonian_m_pauli = hamiltonian_m_pauli

        self.hamiltonian_el_pauli = hamiltonian_el_pauli
        self.hamiltonian_mag_pauli = hamiltonian_mag_pauli

    # others
    @staticmethod
    def str_to_tens(string: str):
        """Transforms bitstring to tensor '0'=[1,0] and '1'=[0,1]
        for example: '00' = array([1, 0, 0, 0]) , '01' = array([0, 1, 0, 0]),
        '10' = array([0, 0, 1, 0]) etc.

        Parameters
        ----------
        string: sequence of 0s and 1s

        Returns
        -------
        array, result of the tensor product

        """

        return reduce(
            lambda x, y: np.kron(x, y), [[1, 0] if x == "0" else [0, 1] for x in string]
        )

    def check_gauss(self, eigenstate: str):
        """From input (eigenstate as bitstring), it returns the set of values
        that the electric fields and the charges have acting on the eigenstate.
        If unphysical, gives KeyError.

        Parameters
        ----------
        eigenstate: string of 0s and 1s

        Returns
        -------
        e_op_sol:

        charge_sol:

        """

        gray_dict = {
            "{0:0{1}b}".format(i ^ (i >> 1), self._n_qubits_g()): k
            for i, k in zip(
                range(2 * self.l_par + 1), range(-self.l_par, self.l_par + 1)
            )
        }

        e_op_sol = [
            (
                j[0],
                gray_dict[
                    eigenstate[::-1][
                        self._n_qubits_g() * i : self._n_qubits_g() * (1 + i)
                    ][::-1]
                ],
            )
            for i, j in enumerate(self.e_op_field_subs)
        ]

        s_p = 0.5 * (self.I - self.Z)  # JW dependent

        charge_sol = [
            (
                q[0],
                (
                    HamiltonianQED.str_to_tens(k)
                    @ s_p.to_matrix()
                    @ HamiltonianQED.str_to_tens(k)
                    - 0.5 * (1 + (-1) ** (n_tmp))
                ).real,
            )
            for q, k, n_tmp in zip(
                self.q_charges_subs,
                eigenstate[::-1][self._n_qubits_g() * len(self.e_op_field_subs) :],
                range(1, len(self.q_charges_subs) + 1),
            )
        ]

        return e_op_sol if self.puregauge else e_op_sol + charge_sol

    def hamiltonian_suppr(
        self,
    ):
        """Suppression Hamiltonian"""
        # Unphysical space suppressors:
        s_down = 0.5 * (self.I + self.Z)  # project to 0
        s_up = 0.5 * (self.I - self.Z)  # project to 1

        if self.encoding == "gray":
            gauge = self.tensor_prod(
                self.I, (self._n_qubits_g() * (self.len_u_op))
            )  # Gray encoding for E fields
        elif self.encoding == "ed":  # exact diagonaliz. dimensions of gauge fields 2l+1
            gauge = sparse.eye((2 * self.l_par + 1) ** (self.len_u_op), format="csr")

        # ******* gauge
        if (
            self.len_u_op > 0 and self.encoding == "gray"
        ):  # only for gray encoding exlcusion of unphyisical states
            h_s = 0
            # the state is projected onto the UNphysical state
            for i in range(2 * self.l_par + 1, 2 ** self._n_qubits_g()):
                gray_str = "{0:0{1}b}".format(i ^ (i >> 1), self._n_qubits_g())
                h_s += reduce(
                    lambda x, y: (x) ^ (y),
                    [s_down if x == "0" else s_up for x in gray_str],
                )

            suppr1 = h_s
            hamiltonian_gauge_suppr = 0.0 * gauge

            for i in range(1, self.len_u_op + 1):
                hamiltonian_gauge_suppr += HamiltonianQED.pauli_tns(
                    self.tensor_prod(
                        self.I, (self._n_qubits_g() * (self.len_u_op - i))
                    ),
                    (suppr1),
                    self.tensor_prod(self.I, (self._n_qubits_g() * (i - 1))),
                ).simplify()

        elif self.len_u_op > 0 and self.encoding == "ed":  # only for ED encoding
            hamiltonian_gauge_suppr = 0.0 * gauge
        else:  # no gauge fields
            hamiltonian_gauge_suppr = 0.0 * self.tensor_prod(
                self.I, int(self.lattice.n_sitestot)
            )

        # ****** fermion
        suppr_f = self.tensor_prod(self.I, (int(self.lattice.n_sitestot)))
        # the state is projected onto zero-charge state (fermions), same number of 1 and 0
        for i in range(2 ** int(self.lattice.n_sitestot)):
            bincount = sum([1 for el in bin(i)[2:] if el == "1"])
            if bincount == int(self.lattice.n_sitestot) / 2:
                binc = format(i, "0%db" % int(self.lattice.n_sitestot))
                suppr_f += -1.0 * reduce(
                    lambda x, y: (x) ^ (y), [s_down if x == "0" else s_up for x in binc]
                )

        hamiltonian_nzcharge_suppr = HamiltonianQED.pauli_tns(suppr_f, gauge)

        if (
            self.tn_comparison
        ):  # TODO: only for 2+1 QED and gray encoding global term in H (possible barren plateaus!)
            # gauss
            gray_physical = list(
                set(
                    [
                        "{:0{width}b}".format(i ^ (i >> 1), width=self._n_qubits_g())
                        for i in range(0, 2 ** (self._n_qubits_g()))
                    ]
                ).difference(
                    [
                        "{:0{width}b}".format(i ^ (i >> 1), width=self._n_qubits_g())
                        for i in range(2 * self.l_par + 1, 2 ** (self._n_qubits_g()))
                    ]
                )
            )

            gray_physical = [
                "".join(i) for i in product(gray_physical, repeat=self.len_u_op)
            ]
            if self.puregauge:
                charge0_physical = []
                phys_state_list = gray_physical
            else:
                charge0_physical = [
                    "".join(i)
                    for i in set(
                        permutations(
                            "0" * (int(self.lattice.n_sitestot) // 2)
                            + "1" * (int(self.lattice.n_sitestot) // 2)
                        )
                    )
                ]  # charge 0 for n_sitestot fermions
                phys_state_list = [
                    "".join([a, b]) for a in charge0_physical for b in gray_physical
                ]

            # check Gauss law for additional unphysical states
            unphys_list_gauss = []

            for eigenstate in phys_state_list:
                sol_gauss_system = solve(
                    [eq.subs(self.check_gauss(eigenstate)) for eq in self.list_gauss],
                    dict=True,
                )[0]
                for j in sol_gauss_system.values():  # if a sol. of systen is >l
                    if j not in range(-self.l_par, self.l_par + 1):
                        unphys_list_gauss.append(eigenstate)

            unphys_list_gauss = list(set(unphys_list_gauss))

            suppr_gaus = 0
            # the state is projected onto the UNphysical state
            for gray_str in unphys_list_gauss:
                suppr_gaus += reduce(
                    lambda x, y: (x) ^ (y),
                    [s_down if x == "0" else s_up for x in gray_str],
                )

            hamiltonian_gauss_suppr = suppr_gaus

        elif self.puregauge:
            hamiltonian_gauss_suppr = 0.0 * gauge
        else:
            hamiltonian_gauss_suppr = 0.0 * HamiltonianQED.pauli_tns(
                self.tensor_prod(self.I, int(self.lattice.n_sitestot)), gauge
            )

        if self.puregauge:
            hamiltonian_suppress = (hamiltonian_gauge_suppr) + (hamiltonian_gauss_suppr)
        elif self.len_u_op > 0:
            hamiltonian_suppress = (
                HamiltonianQED.pauli_tns(
                    self.tensor_prod(self.I, int(self.lattice.n_sitestot)),
                    hamiltonian_gauge_suppr,
                )
                + (hamiltonian_nzcharge_suppr)
                + (hamiltonian_gauss_suppr)
            )
        else:  # no gauge fields
            hamiltonian_suppress = hamiltonian_nzcharge_suppr
        if isinstance(
            hamiltonian_suppress,
            (np.ndarray, sparse._csr.csr_matrix, sparse._coo.coo_matrix),
        ):
            self.hamiltonian_suppress = hamiltonian_suppress
        else:
            self.hamiltonian_suppress = hamiltonian_suppress.to_matrix(sparse=True)
