#NEW VERSION WITH ONLY HAMILTONIAN
#
#
"""Definition of the Hamiltonian for QED lattice NxN"""
from __future__ import annotations
import math
import warnings
from functools import reduce
import time
from itertools import permutations, product,combinations
import re
from typing import List
import numpy as np
import networkx as nx
from networkx import all_simple_paths, get_edge_attributes
from networkx.generators.classic import empty_graph
from networkx.utils import pairwise
from qiskit.opflow import Z, X, Y, I, PauliSumOp,OperatorBase
from qiskit.quantum_info import SparsePauliOp
from IPython.display import display
from scipy import special as sp
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, solve, lambdify, Mul, Eq, latex
from sympy.physics.quantum.dagger import Dagger

#from HC_Lattice import HCLattice

class HamiltonianQED():#HCLattice):

    """The algorithm computes the expression of the Quantum Electrodynamics (QED)
    Kogut-Susskind Hamiltonian,
    both in terms of sympy.symbols and in qiskit Pauli matrices for 2-D and
    1-D lattices.
    The latter formulation is suitable for quantum circuits.
    For fermionic degrees of freedom, the Jordan-Wigner transformation is applied.
    The discretisation of the group U(1) is done by means of group of integer numbers
    Z_(2L+1).In this definition, the Gray encoding is applied for the gauge fields.

    From an instance of number of sites in the lattice, in the horizontal (x) and
    vertical (y) direction,
    and the boundary condition, the code generates the Hamiltonian related to that
    lattice.

    The final expression of the Hamiltonian is given in terms of Pauli matrices,
    and it is written by
    following the order right-left, up-down, i.e. the first term is the one acting
    on the rightmost site.


    Parameters
    ----------

    nx_sites,ny_sites: int
            Number of sites in the lattice in the x and y direction.

    g: float or int
        Coupling of the theory.

    fact_e_op:float or int
        Factor in front of the electric Hamiltonian.

    fact_b_op:float or int
        Factor in front of the magnetic Hamiltonian.

    m:float or int
        Mass term in the Hamiltonian.

    omega:float or int
        Factor in front of the kinetic Hamiltonian.

    l: int
        Truncation parameter. Defines how many values the gauge fields take,
        e.g. l=1 -> ±1,0 .

    ll: int
        Discretisation parameter L.

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

    ksphase: bool
        Phase in the kientic Hamiltonian, option for tests. Muste be kept True for
        Kogut-Susskind Hamiltonian.


    display_hamiltonian: bool
        If True, the Hamiltonian and the Gauss law equations are displayed in the output.

    lambd=int or float
        Parameter for the suppression factor in the Hamiltonian.

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
        magnetic_basis: bool = False,
        pbc: bool = False,
        puregauge: bool = False,
        static_charges_values: dict | None = None,
        e_op_out_plus: bool = False,
        ksphase: bool = True,
        display_hamiltonian: bool = False,
        tn_comparison: bool = False,
    ) -> None:
        self.n_sites = n_sites
        self.pbc = pbc

        #super().__init__(n_sites,pbc)
        self.lattice = lattice
        self.l_par = l
        self.ll_par = ll
        self.magnetic_basis = magnetic_basis
        self.puregauge = puregauge
        self.static_charges_values = static_charges_values
        self.e_op_out_plus = e_op_out_plus
        self.ksphase = ksphase
        self.display_hamiltonian = display_hamiltonian
        self.tn_comparison = tn_comparison

        print("HamiltonianQED: Initializing...")


        self._symlist = ["I", "X", "Y", "Z", "Sd", "S-", "Su", "S+"]
        self.alpha = 2 * np.pi / (2 * self.ll_par + 1) if self.magnetic_basis else 0

        # get the start time
        start_time = time.time()

        #list of dynamical and static charges
        self.str_node_f = lambda node: str(node) if self.lattice.dims==1 else "".join(map(str, node))

        self.q_charge_str_list = ["q_" +self.str_node_f(node)  for node in self.lattice.graph.nodes if self.puregauge is False]
        self.static_charges_str_list = [
                "Q_" +self.str_node_f(node)  for node in self.lattice.graph.nodes
                if  self.static_charges_values is not None
            ]
        # Dictionary of all elements (E and charges q) in the system with their symbols
        self.e_op_dict = {
            s_tmp: symbols(s_tmp)
            for s_tmp in self.lattice.list_edges2_e_op
            + self.q_charge_str_list
            + self.static_charges_str_list
        }

        self.u_op_dict = {
            s_tmp: symbols(s_tmp)
            for s_tmp in self.lattice.list_edges2_u_op

        }

        # #Gauss law equations in a list and display them
        self.gauss_equations()
        if self.display_hamiltonian:
            print(">> Gauss law system of equations (symbolic + latex):")
            [display(Eq(i, 0)) for i in self.list_gauss]
            [print(latex(i) + " &= 0 \\\\ \\nonumber") for i in self.list_gauss[:-1]]
            print(latex(self.list_gauss[-1]) + " &= 0", "\n")

        #Solution of gauss law equations
        self.sol_gauss = solve(self.list_gauss, dict=True)[0]


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
                    for j, k in zip(self.lattice.list_edges2_e_op, self.lattice.list_edges2_u_op)
                ]
            )
            for k in self.e_op_free
        ]
        self.u_op_free_dag = [
            k.subs(
                [
                    (symbols(j), Symbol(k + "D"))
                    for j, k in zip(self.lattice.list_edges2_e_op, self.lattice.list_edges2_u_op)
                ]
            )
            for k in self.e_op_free
        ]  # U^dag

        # length of e_op_free and u_op_free
        self.len_e_op = len(self.e_op_free)
        self.len_u_op = len(self.u_op_free)



        # Define the espressions for substituting symbols into Pauli strings
        self._symbol_to_pauli()
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


    def get_hamiltonian(self,g_var=1.0, m_var=1.0, omega=1.0,fact_b_op=1.0,fact_e_op=1.0,lambd=1000.):
        """ Returns the Hamiltonian of the system """

        #Hamiltonian for fermions
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
            self.hamiltonian_gauge + self.hamiltonian_ferm + lambd* self.hamiltonian_suppress
        ).reduce()

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
                ga_tmp=-1*self.e_op_dict['q_'+self.str_node_f(node)]
                gc_tmp += self.e_op_dict['q_'+self.str_node_f(node)]
            if self.static_charges_values is not None:
                ga_tmp-=1*self.e_op_dict['Q_'+self.str_node_f(node)]
                gc_tmp+=self.e_op_dict['Q_'+self.str_node_f(node)]

            e_op_i = "E_" +self.str_node_f(node)
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

    @staticmethod
    def _subs_hamilt_sym_to_pauli(hamilt_sym: list, subst: list):
        """Function that takes as input list like [U_00x, 1, U_01xd, U_00yd],
        then applies substitutions rules (in subst) and combine elements
        with @ operation between Pauli matrices.
        If type(input)!=symbol then use * operator.
        Otherwise, for Pauli matrix multiplication @ is needed.

        Returns a list of symbols for mass Hamiltonian of the type 'phi^dag phi'.


        """
        hamilt_func = lambdify(list(zip(*subst))[0], hamilt_sym)
        hamilt_input = hamilt_func(*list(zip(*subst))[1])

        container = (
            hamilt_input[0] * I
            if isinstance(hamilt_input[0],(int,float,complex))
            else hamilt_input[0]
        )

        for i in hamilt_input[1:]:
            if not isinstance(i, (int,float,complex)):
                container @= i
            else:
                container *= i

        return container

    @staticmethod
    def jw_func(n_tmp: int, n_qubits: int):
        """Jordan-Wigner for 2 terms phi^dag, phi

        Inputs:
            n_tmp: index of fermionic operator

            n_qubits: n.er of total qubits in the string

        """
        sgm = PauliSumOp( SparsePauliOp.from_sparse_list( [ ( "X", [ 0, ], 0.5, ), ] + [ ( "Y", [ 0, ], (-0.5j), ), ], num_qubits=1, ) )
        sgp = PauliSumOp( SparsePauliOp.from_sparse_list( [ ( "X", [ 0, ], 0.5, ), ] + [ ( "Y", [ 0, ], (0.5j), ), ], num_qubits=1, ) )

        assert n_tmp > 0
        if n_tmp == 1:
            jw_dagk = (I) ^ 0
            jwk = (I) ^ 0

        else:
            jw_dagk = ((1j) ** (n_tmp - 1)) * (Z ^ (n_tmp - 1))
            jwk = ((-1j) ** (n_tmp - 1)) * (Z ^ (n_tmp - 1))

        jw_dag = (I ^ (n_qubits - n_tmp)) ^ (sgm) ^ (jw_dagk)
        jw_nodag = (I ^ (n_qubits - n_tmp)) ^ (sgp) ^ (jwk)

        return jw_dag, jw_nodag  # then use: jw_dag@jw_nodag for phi^dag phi

    # ######utilities and operators
    @staticmethod
    def _str_to_pauli(lst, n_tmp):
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

            sparse_prod = I ^ (n_tmp)
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
            pauli_sum = PauliSumOp(sparse_prod, coeff=complex(facts[0]))

            pauli_res += pauli_sum

        return pauli_res.reduce()

    # ##ENCODING FUNCTIONS AND OPERATORS
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
                        *HamiltonianQED._trans_map(self._gray_map()[st_fact],
                                                   self._gray_map()[st_fact]),
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
                        u_op_list.append([fact, *HamiltonianQED._trans_map(e1_tmp, e2_tmp)])
        u_oper = [
            f"{v_elem[0]} "
            + " ".join(
                f"{self._symlist[s_tmp]}_{self._n_qubits_g()-i-1}"
                for i, s_tmp in enumerate(v_elem[1:])
            )
            for v_elem in u_op_list
        ]

        return u_oper

    # electric field operator
    def _e_operator(self, index: int = 1):
        """Electric field operator
        Input arguments:
            index = on which qubits the operator acts
            tot_fiels = how many fields are in the Hamiltonian
            magnetic_basis = magnetic basis (default True)
        """
        # When fermions add I^(n.er fermions) in the hamiltonian

        if self.magnetic_basis is False:  # electric basis
            return (
                (I ^ (self._n_qubits_g() * (self.len_e_op - index)))
                ^ HamiltonianQED._str_to_pauli(self._r_c()[1], self._n_qubits_g())
                ^ (I ^ (self._n_qubits_g() * (index - 1)))
            )

        else:  # TODO magnetic basis
            e_op = 0

            _l_c_p = (
                (I ^ (self._n_qubits_g() * (self.len_e_op - index)))
                ^ HamiltonianQED._str_to_pauli(self._l_c(), self._n_qubits_g())
                ^ (I ^ (self._n_qubits_g() * (index - 1)))
            )
            _l_c_m = _l_c_p.adjoint()

            lcm_nu = [
                _l_c_m,
            ]
            lcp_nu = [
                _l_c_p,
            ]

            for nu in range(1, 2 * self.ll_par + 1):
                fnuS_func = float(
                    (-1) ** (nu + 1)
                    / (2 * math.pi)
                    * (
                        sp.polygamma(
                            0, (2 * self.ll_par + 1 + nu) / (2 * (2 * self.ll_par + 1))
                        )
                        - sp.polygamma(0, nu / (2 * (2 * self.ll_par + 1)))
                    )
                )
                e_op_factor = (fnuS_func / (2j) * (lcm_nu[-1] - lcp_nu[-1])).reduce()

                if nu < 2 * self.ll_par:
                    lcm_nu.append((lcm_nu[-1] @ _l_c_m).reduce())
                    lcp_nu.append((lcp_nu[-1] @ _l_c_p).reduce())

                if [0.0 + 0.0j] in e_op_factor.coeffs:
                    break
                else:
                    e_op += e_op_factor

            return e_op

    # link operator
    def _u_operator(self, index: int = 1):
        """Link operator U
        Input arguments: index = on which qubits the operator acts
        """
        # _n_qubits_g valid for _gray_map, could be different for other encodings

        if self.magnetic_basis is False:  # electric basis
            return (
                (I ^ (self._n_qubits_g() * (self.len_u_op - index)))
                ^ (HamiltonianQED._str_to_pauli(self._l_c(), self._n_qubits_g()))
                ^ (I ^ (self._n_qubits_g() * (index - 1)))
            ).reduce()

        else:  # magnetic basis
            return  # self.eiSz_noidx_1() #TODO Start to fix for magnetic basis

    # # squared electric field ( E^2)
    # def E_op2(self, index: int = 1):
    #     """Electric field operator ^2
    #     Input arguments:
    #         index = on which qubits the operator acts
    #         tot_fiels = how many fields are in the Hamiltonian
    #         magnetic_basis = magnetic basis (default True)
    #     """

    #     if self.magnetic_basis is False:  # electric basis
    #         return (
    #             (I ^ (self._n_qubits_g() * (tot_fields - index)))
    #             ^ (HamiltonianQED._str_to_pauli(self._r_c()[1], self._n_qubits_g()) ** 2)
    #             ^ (I ^ (self._n_qubits_g() * (index - 1)))
    #         ).reduce()

    #     else:  # magnetic basis
    #         e_op = 0
    #         _l_c_m = (
    #             (I ^ (self._n_qubits_g() * (tot_fields - index)))
    #             ^ HamiltonianQED._str_to_pauli(self._l_c(), self._n_qubits_g())
    #             ^ (I ^ (self._n_qubits_g() * (index - 1)))
    #         )
    #         _l_c_p = _l_c_m.adjoint()

    #         lcm_nu = [
    #             _l_c_m,
    #         ]
    #         lcp_nu = [
    #             _l_c_p,
    #         ]

    #         for nu in range(1, 2 * self.ll_par + 1):
    #             fnuC = float(
    #                 (-1) ** (nu)
    #                 / (4 * np.pi**2)
    #                 * (
    #                     sp.polygamma(1, nu / (2 * (2 * self.ll_par + 1)))
    #            - sp.polygamma(1, (2 * self.ll_par + 1 + nu) / (2 * (2 * self.ll_par + 1)))
    #                 )
    #             )
    #             e_op_factor = (fnuC / (2) * (lcm_nu[-1] + lcp_nu[-1])).reduce()

    #             if nu < 2 * self.ll_par:
    #                 lcm_nu.append((lcm_nu[-1] @ _l_c_m)).reduce()
    #                 lcp_nu.append((lcp_nu[-1] @ _l_c_p)).reduce()

    #             if [0.0 + 0.0j] in e_op_factor.coeffs:
    #                 break
    #             else:
    #                 e_op += e_op_factor

    #         return e_op + 1 / 3 * self.ll_par * (self.ll_par + 1) *
    # (I ^ (self._n_qubits_g() * (tot_fields)))
    # ### exp and cosine functions

    # def eiSz_noidx_1(self):
    #     """Exponential function for Sz
    #     Input args.: alpha - numerical coefficient"""
    #     _n_qubits_g = int(np.ceil(np.log2(2 * self.l_par + 1)))

    #     sznoidx = _r_c()[0]
    #     projterm = f"{1.0} " + " ".join(sznoidx[0][1])
    #     arg =  HamiltonianQED._str_to_pauli(["1 I_0"], _n_qubits_g) + (
    #         (math.cos(alpha * sznoidx[0][0]) + 1j * math.sin(alpha * sznoidx[0][0]) - 1)
    #         *  HamiltonianQED._str_to_pauli([projterm], _n_qubits_g)
    #     )

    #     for nn in range(1, 2 * self.l_par ):
    #         aux = sznoidx[nn]
    #         projterm = f"{1.0} " + " ".join(aux[1])
    #         arg = (
    #             (
    #                  HamiltonianQED._str_to_pauli(["1 I_0"], _n_qubits_g)
    #                 + (
    #                     (math.cos(alpha * aux[0]) + 1j * math.sin(alpha * aux[0]) - 1)
    #                     *  HamiltonianQED._str_to_pauli([projterm], _n_qubits_g)
    #                 )
    #             )
    #             @ arg
    #         ).reduce()

    #     return arg

    # re_psum = lambda op: 0.5 * (op + op.adjoint())
    # im_psum = lambda op: -0.5j * (op - op.adjoint())

    # def cosSz_noidx_1(self):
    #     """Cosine function for Sz
    #     Input args.: alpha - numerical coefficient"""
    #     return (re_psum(self.eiSz_noidx_1())).reduce()

    # def cosSz_noidx_3(self):
    #     """Cosine function for sum of 3 Sz (3 rotators system) gauge theory (w/o fermions)
    #     Input args.: alpha - numerical coefficient"""
    #     ei1 = self.eiSz_noidx_1()
    #     r1 = (re_psum(ei1)).reduce()
    #     i1 = (im_psum(ei1)).reduce()
    #     arg = r1 ^ r1 ^ r1
    #     arg -= r1 ^ i1 ^ i1
    #     arg -= i1 ^ r1 ^ i1
    #     arg -= i1 ^ i1 ^ r1
    #     return arg

    # def cos_Uop(self,operators):
    #     """Cosine function for generic espression
    #     Input arg: list of operators. See e.g. H_b_5links_ferm"""
    #     arg = reduce(lambda x, y: x ^ y, operators)

    #     return (0.5 * (arg + arg.adjoint())).reduce()

    # TODO end to fix

    def _symbol_to_pauli(self):
        """Converts a string of symbols into a Pauli operator"""
        # ****GAUGE PART

        # Static charges
        if self.static_charges_values:
            static_charges_list = [
                self.static_charges_values[k]
                if k in self.static_charges_values.keys()
                else 0
                for k in self.lattice.jw_sites
            ]
            static_charges_subs = [
                (
                    symbols("Q_" + "".join(map(str, k))),
                    j * (I ^ (int(self.lattice.n_sitestot) + self._n_qubits_g() * self.len_e_op)),
                )
                for k, j in zip(self.lattice.jw_sites, static_charges_list)
            ]
        else:
            static_charges_subs = []

        # E FIELD
        q10 = -0.5 * (I + Z)  # JW dependent
        q00 = 0.5 * (I - Z)

        # E field in terms of Pauli matrices
        _e_op_elem = lambda i: self._e_operator(index=i + 1)

        if self.puregauge:
            e_op_field_subs = [(s_tmp, _e_op_elem(i)) for i, s_tmp in enumerate(self.e_op_free)]
            q_charges_subs = []
        else:
            e_op_field_subs = [
                (s_tmp, ((I ^ (int(self.lattice.n_sitestot))) ^ (_e_op_elem(i))))
                for i, s_tmp in enumerate(self.e_op_free)
            ]
            #charge operator in terms of Pauli matrices
            q_el = (
                lambda i, q: (I ^ (int(self.lattice.n_sitestot) - 1 - i))
                ^ (q)
                ^ (I ^ (self._n_qubits_g() * self.len_e_op + i))
            )
            sum_k = lambda k: k if self.lattice.dims ==1 else sum(k)
            q_charges_subs = [
                (
                    symbols("q_" +self.str_node_f(k)),
                    q_el(i, q10),
                )
                if sum_k(k) % 2
                else (
                    symbols("q_" +self.str_node_f(k)),
                    q_el(i, q00),
                )
                for i, k in enumerate(self.lattice.jw_sites)
            ] + static_charges_subs

        # U field in terms of Pauli matrices
        _u_op_elem = lambda i: self._u_operator(index=i + 1)

        if self.puregauge:
            u_op_field_subs = [(s_tmp, _u_op_elem(i)) for i, s_tmp in enumerate(self.u_op_free)] + \
            [
                (
                    s_tmp,
                    (self._u_operator(index=i + 1)).adjoint(),
                )
                for i, s_tmp in enumerate(self.u_op_free_dag)
            ]
        else:
            u_op_field_subs = [
                (s_tmp, (I ^ int(self.lattice.n_sitestot)) ^ _u_op_elem(i))
                for i, s_tmp in enumerate(self.u_op_free)
            ] + [
                (s_tmp, ((I ^ int(self.lattice.n_sitestot)) ^ _u_op_elem(i)).adjoint())
                for i, s_tmp in enumerate(self.u_op_free_dag)
            ]

        # ****list of Pauli substitutions for fermionic sites
        if self.puregauge:
            phi_jw_subs = []
        else:
            phi_el = lambda i, j: (HamiltonianQED.jw_func(i + 1, int(self.lattice.n_sitestot))[j]) ^ (
                I ^ (self._n_qubits_g() * self.len_u_op)
            )

            phi_jw_subs = [
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

        self.e_op_field_subs = e_op_field_subs
        self.q_charges_subs = q_charges_subs
        self.u_op_field_subs = u_op_field_subs
        self.phi_jw_subs = phi_jw_subs


    # HAMILTONIAN
    # * symbols
    # Define Hamiltonian and apply Gauss laws
    def _hamiltonian_el_autom(self):
        """Hamiltonian for E field"""
        hamiltonian_el_sym = [Symbol(str(s)) for s in self.lattice.list_edges2_e_op]
        hamiltonian_el_sym = sum(
            [
                x**2 if x not in self.sol_gauss else (self.sol_gauss[x]) ** 2
                for x in hamiltonian_el_sym
            ]
        )  # Gauss law applied
        self.hamiltonian_el_sym = hamiltonian_el_sym

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
                phase=1
                hamiltonian_k_sym.append((phase, jw_dict[i[0]][0], hamilt_k_elem, jw_dict[i[1]][1]))

            elif self.lattice.dims == 2:

                phase = (
                    (-1) ** (sum(i[0]) % 2) if self.ksphase and i[0][1] != i[1][1] else 1
                )  # change in y direction if x is odd
                xy_term = 'y' if i[0][1] != i[1][1] else 'x' #if x - adjoint, if y + adjoint

                hamiltonian_k_sym.append((xy_term,phase, jw_dict[i[0]][0], hamilt_k_elem, jw_dict[i[1]][1]))

            elif self.lattice.dims == 3:
                if self.ksphase:
                    #x-direction
                    if i[0][0] != i[1][0]:
                        phase = 1
                    #y-direction
                    elif i[0][1] != i[1][1]:
                        phase = (-1)**(sum(i[0][:2]+1)%2)
                    #z-direction
                    elif i[0][2] != i[1][2]:
                        phase = (-1)**(sum(i[0][:2])%2)
                else:
                    phase = 1 #y or z direction

                i_term = 'x' if i[0][0] != i[1][0] else 'y' if i[0][1] != i[1][1] else 'z' if i[0][2] != i[1][2] else None

                hamiltonian_k_sym.append((i_term,phase, jw_dict[i[0]][0], hamilt_k_elem, jw_dict[i[1]][1])) #phi^dag U phi

            else:
                raise ValueError("Only 1, 2 and 3 dimensions are supported.")


        self.hamiltonian_k_sym = hamiltonian_k_sym

    # build H
    def build_hamiltonian_tot(self):  # TODO: printed latex expressions too long
        """Builds the total Hamiltonian of the system."""
        # ************************************  H_E   ************************************
        if self.len_e_op > 0:
            # Pauli expression
            hamiltonian_el_func = lambdify(
                list(zip(*self.q_charges_subs + self.e_op_field_subs))[0],
                self.hamiltonian_el_sym,
            )

            hamiltonian_el_pauli = (
                0.5
                * hamiltonian_el_func(
                    *list(zip(*self.q_charges_subs + self.e_op_field_subs))[1]
                ).reduce()
            )  # (must be then multiplied by g^2)

            if self.display_hamiltonian:  # Hamiltonian to print
                display_hamiltonian_el = Eq(
                    Symbol("H_E"), (Symbol("g") ** 2) / 2 * self.hamiltonian_el_sym
                )
                display(display_hamiltonian_el)
                print(latex(display_hamiltonian_el))
        else:  # no gauge fields (e.g. 1d OBC case)
            hamiltonian_el_pauli = 0.0 * (
                I ^ (int(self.lattice.n_sitestot) + self._n_qubits_g() * (self.len_u_op))
            )

        # ************************************  H_B   ************************************
        if len(self.u_op_free) > 0 and self.lattice.dims > 1:
            # Pauli expression
            hamiltonian_mag_sym = sum(
                [
                    HamiltonianQED._subs_hamilt_sym_to_pauli(h, self.u_op_field_subs)
                    for h in self.hamiltonian_mag_subs
                ]
            )
            hamiltonian_mag_pauli = (
                float(0.5) * (hamiltonian_mag_sym + hamiltonian_mag_sym.adjoint())
            ).reduce()  # (must be then multiplied by -1/g^2)
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
            hamiltonian_mag_pauli = 0.0 * (
                I ^ (int(self.lattice.n_sitestot) + self._n_qubits_g() * (self.len_u_op))
            )
        # ************************************  H_K   ************************************
        # Pauli expression
        if self.lattice.dims == 1:
            hamiltonian_k_1x = sum(
            [
                HamiltonianQED._subs_hamilt_sym_to_pauli(h, self.u_op_field_subs + self.phi_jw_subs)
                for h in self.hamiltonian_k_sym
            ]
            )

            hamiltonian_k_pauli = (
                0.5j * (hamiltonian_k_1x - hamiltonian_k_1x.adjoint())
            ).reduce()  # (must be then multiplied by omega)

        elif self.lattice.dims == 2:

            hamiltonian_k_1x = sum(
                [
                    HamiltonianQED._subs_hamilt_sym_to_pauli(h[1:], self.u_op_field_subs + self.phi_jw_subs)
                    for h in self.hamiltonian_k_sym if h[0]=='x'
                ]
            )
            hamiltonian_k_1y = sum(
                [
                    HamiltonianQED._subs_hamilt_sym_to_pauli(h[1:], self.u_op_field_subs + self.phi_jw_subs)
                    for h in self.hamiltonian_k_sym if h[0]=='y'
                ]
            )

            hamiltonian_k_pauli = (
                0.5j * (hamiltonian_k_1x - hamiltonian_k_1x.adjoint()) - 0.5 * (hamiltonian_k_1y + hamiltonian_k_1y.adjoint())
            ).reduce()  # (must be then multiplied by omega)

        elif self.lattice.dims == 3:
            hamiltonian_k_1x = sum(
                [
                    HamiltonianQED._subs_hamilt_sym_to_pauli(h[1:], self.u_op_field_subs + self.phi_jw_subs)
                    for h in self.hamiltonian_k_sym if h[0]=='x'
                ]
            )
            hamiltonian_k_1y = sum(
                [
                    HamiltonianQED._subs_hamilt_sym_to_pauli(h[1:], self.u_op_field_subs + self.phi_jw_subs)
                    for h in self.hamiltonian_k_sym if h[0]=='y'
                ]
            )
            hamiltonian_k_1z = sum(
                [
                    HamiltonianQED._subs_hamilt_sym_to_pauli(h[1:], self.u_op_field_subs + self.phi_jw_subs)
                    for h in self.hamiltonian_k_sym if h[0]=='z'
                ]
            )
            hamiltonian_k_pauli = (
                0.5j * (hamiltonian_k_1x - hamiltonian_k_1x.adjoint()) - 0.5 * (hamiltonian_k_1y + hamiltonian_k_1y.adjoint())
                + 0.5j * (hamiltonian_k_1z - hamiltonian_k_1z.adjoint())
            ).reduce()  # (must be then multiplied by omega)

        else:
            raise ValueError("Dimension not supported")

        if self.display_hamiltonian:#TODO 1d
            # Hamiltonian to print

            if self.lattice.dims == 1:
                hamiltonian_k_display = [
                (
                    k[0],
                    Dagger(Symbol(str(k[1])[:-1], commutative=False)),
                    Dagger(Symbol(str(k[2])[:-1], commutative=False)),
                    k[3],
                )
                if str(k[2])[-1] == "d"
                else (
                    k[0],
                    Dagger(Symbol(str(k[1])[:-1], commutative=False)),
                    k[2],
                    k[3],
                )
                for k in self.hamiltonian_k_sym
            ]

                display_hamiltonian_k = Eq(
                Symbol("H_K"),
                (Symbol("Omega")*1j / 2)
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
                hamiltonian_k_display = [
                    (
                        k[1],
                        Dagger(Symbol(str(k[2])[:-1], commutative=False)),
                        Dagger(Symbol(str(k[3])[:-1], commutative=False)),
                        k[4],
                    )
                    if str(k[3])[-1] == "D"
                    else (
                        k[1],
                        Dagger(Symbol(str(k[2])[:-1], commutative=False)),
                        k[3],
                        k[4],
                    )
                    for k in self.hamiltonian_k_sym
                ]
                h_k_x_disp = 0
                h_k_y_disp = 0
                h_k_z_disp = 0
                for k,j in zip(hamiltonian_k_display,self.hamiltonian_k_sym):
                    if j[0]=='x':
                        h_k_x_disp+=  sum( [ Mul(*k, evaluate=False) if k[2] != 1 else Mul(*k) ] )
                    elif j[0]=='y':
                        h_k_y_disp+= sum( [ Mul(*k, evaluate=False) if k[2] != 1 else Mul(*k) ] )
                    elif j[0]=='z':
                        h_k_z_disp+= sum( [ Mul(*k, evaluate=False) if k[2] != 1 else Mul(*k) ] )

                if self.lattice.dims == 3:
                    h_k_z = 0.5j*(h_k_z_disp-Symbol("h.c.(z)", commutative=False))
                else:
                    h_k_z = 0
                display_hamiltonian_k= Eq(
                                Symbol("H_K"),
                                (Symbol("Omega") )
                                * (0.5j*(h_k_x_disp-Symbol("h.c.(x)", commutative=False)) - 0.5*(h_k_y_disp+Symbol("h.c.(y)", commutative=False))
                                + h_k_z),
                                evaluate=False,
                            )

            display(display_hamiltonian_k)
            print(latex(display_hamiltonian_k))
        # ************************************  H_M   ************************************
        # H_M in terms of Paulis
        hamiltonian_m_pauli = sum(
            [
                (-1) ** j * HamiltonianQED._subs_hamilt_sym_to_pauli(h, self.phi_jw_subs)
                for j, h in enumerate(self.hamiltonian_m_sym)
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

        self.hamiltonian_el_pauli = hamiltonian_el_pauli
        self.hamiltonian_mag_pauli = hamiltonian_mag_pauli
        self.hamiltonian_k_pauli = hamiltonian_k_pauli
        self.hamiltonian_m_pauli = hamiltonian_m_pauli


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

        s_p = 0.5 * (I - Z)  # JW dependent

        charge_sol = [
            (
                q[0],
                (
                    HamiltonianQED.str_to_tens(k) @ s_p.to_matrix() @ HamiltonianQED.str_to_tens(k)
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
        s_down = 0.5 * (I + Z)  # project to 0
        s_up = 0.5 * (I - Z)  # project to 1

        # ******* gauge
        if self.len_u_op > 0:
            h_s = 0
            # the state is projected onto the UNphysical state
            for i in range(2 * self.l_par + 1, 2 ** self._n_qubits_g()):
                gray_str = "{0:0{1}b}".format(i ^ (i >> 1), self._n_qubits_g())
                h_s += reduce(
                    lambda x, y: (x) ^ (y), [s_down if x == "0" else s_up for x in gray_str]
                )

            suppr1 = h_s
            hamiltonian_gauge_suppr = 0.0 * (I ^ (self._n_qubits_g() * (self.len_u_op)))

            for i in range(1, self.len_u_op + 1):
                hamiltonian_gauge_suppr += (
                    (I ^ (self._n_qubits_g() * (self.len_u_op - i)))
                    ^ (suppr1)
                    ^ (I ^ (self._n_qubits_g() * (i - 1)))
                ).reduce()

        else:
            hamiltonian_gauge_suppr = 0.0 * (I ^ int(self.lattice.n_sitestot))

        # ****** fermion
        suppr_f = I ^ (int(self.lattice.n_sitestot))
        # the state is projected onto zero-charge state (fermions), same number of 1 and 0
        for i in range(2**int(self.lattice.n_sitestot)):
            bincount = sum([1 for el in bin(i)[2:] if el == "1"])
            if bincount == int(self.lattice.n_sitestot) / 2:
                binc = format(i, "0%db" % int(self.lattice.n_sitestot))
                suppr_f += -1.0 * reduce(
                    lambda x, y: (x) ^ (y), [s_down if x == "0" else s_up for x in binc]
                )

        hamiltonian_nzcharge_suppr = (suppr_f) ^ (I ^ (self._n_qubits_g() * self.len_u_op))


        if self.tn_comparison:  # TODO: only for 2+1 QED
            # gauss #TODO: global term in H (possible barren plateaus!)
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
                            "0" * (int(self.lattice.n_sitestot) // 2) + "1" * (int(self.lattice.n_sitestot) // 2)
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
                    lambda x, y: (x) ^ (y), [s_down if x == "0" else s_up for x in gray_str]
                )

            hamiltonian_gauss_suppr = suppr_gaus

        elif self.puregauge:
            hamiltonian_gauss_suppr = 0.0 * (I ^ (self._n_qubits_g() * (self.len_u_op)))
        else:
            hamiltonian_gauss_suppr = 0.0 * (
                (I ^ int(self.lattice.n_sitestot)) ^ (I ^ (self._n_qubits_g() * (self.len_u_op)))
            )

        if self.puregauge:
            hamiltonian_suppress = (
                (hamiltonian_gauge_suppr) + (hamiltonian_gauss_suppr)
            ).reduce()
        elif self.len_u_op > 0:
            hamiltonian_suppress = (
                ((I ^ int(self.lattice.n_sitestot)) ^ hamiltonian_gauge_suppr)
                + (hamiltonian_nzcharge_suppr)
                + (hamiltonian_gauss_suppr)
            ).reduce()
        else:  # no gauge fields
            hamiltonian_suppress = ((hamiltonian_nzcharge_suppr)).reduce()

        self.hamiltonian_suppress = hamiltonian_suppress



