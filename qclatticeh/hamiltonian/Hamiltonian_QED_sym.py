"""Definition of the Hamiltonian for QED lattice NxN: symbolic expression"""
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
#from scipy import special
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
    sin,
    Sum,
)
from sympy.abc import nu
from sympy.core.numbers import ImaginaryUnit
from sympy.physics.quantum.dagger import Dagger
from scipy.sparse.linalg import eigs
from scipy import sparse
import gc

SPARSE_PAULI = qiskit.quantum_info.operators.symplectic.sparse_pauli_op.SparsePauliOp

# from memory_profiler import profile#, memory_usage


class HamiltonianQED_sym:

    """The algorithm computes the symbolic expression of the Quantum Electrodynamics (QED)
    Kogut-Susskind Hamiltonian, in terms of sympy.symbols for lattices
    from 1D to 3D.
    From an instance of a n-dimensional lattice the code generates the Hamiltonian
    related to that lattice.

    The final expression of the Hamiltonian is given in terms of symbols, which can be
    substituted with numerical values to obtain the numerical Hamiltonian.

    Parameters
    ----------

    config: dict
        Contains the following arguments:
        {
        lattice: class
            Instance of the class Lattice.

        n_sites: list
            Number of sites in each direction.

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

        e_op_free_input: list
            List of free electric fields. If None, then it is computed from the solution of the Gauss law.
            Important for quantum circuits.

        gauss_law: bool
            If True, the Gauss law equations are applied to the Hamiltonian. If False, the Gauss law
            is not applied.
        
        n_flavors: int
            Number of fermionic flavors in the system.
        }

    display_hamiltonian: bool
        If True, the Hamiltonian and the Gauss law equations are displayed in the output.


    """

    def __init__(
        self,
        config: dict,
        display_hamiltonian: bool = False,
    ) -> None:
        # dict config inputs
        self.lattice = config.get("latt")
        self.n_sites = config["n_sites"]
        self.ll_par = config["L"] if "L" in config else 2
        self.encoding = config["encoding"] if "encoding" in config else "gray"
        self.magnetic_basis = config["magnetic_basis"]
        self.pbc = config["pbc"] if "pbc" in config else False
        self.puregauge = config["puregauge"] if "puregauge" in config else False
        self.static_charges_values = (
            config["static_charges_values"]
            if "static_charges_values" in config
            else None
        )
        self.e_op_out_plus = (
            config["e_op_out_plus"] if "e_op_out_plus" in config else True
        )
        self.e_op_free_input = (
            config["e_op_free_input"] if "e_op_free_input" in config else None
        )
        self.gauss_law = (config["gauss_law"] if "gauss_law" in config else True)

        self.n_flavors = config["n_flavors"] if "n_flavors" in config else 1

        # external inputs
        self.display_hamiltonian = display_hamiltonian


        if not self.puregauge and np.prod(self.n_sites) % 2 != 0:
            raise Warning(
                "Attention: total number of staggered fermionic degrees of freedom doesn't respect two-component spinors."
            )

        if self.magnetic_basis and self.lattice.dims != 2:
            raise ValueError("Magnetic basis is only implemented for 2D lattices")

        if self.display_hamiltonian:
            print("Alpha angle \u03B1=2 \u03C0/2L+1")
        self.alpha = 2 * np.pi / (2 * self.ll_par + 1) if self.magnetic_basis else 0

        print("HamiltonianQED_sym: Initializing...")
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
            for node in self.lattice.jw_sites
            if self.puregauge is False
        ]
        self.static_charges_str_list = [
            "Q_" + self.str_node_f(node)
            for node in self.lattice.graph.nodes
            if self.static_charges_values is not None
        ]
        # Dictionary of all elements (E and charges q) in the system with their symbols
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

        if self.gauss_law:
            # #Gauss law equations in a list and display them if links
            self.gauss_equations()
            if self.display_hamiltonian:
                print(">> Gauss law system of equations (symbolic + latex):")
                print(
                    "static charges:",
                    [
                        "Q_" + self.str_node_f(key) + f"={val}"
                        for key, val in self.static_charges_values.items()
                    ]
                    if self.static_charges_values is not None
                    else "None",
                )
                [display(Eq(i, 0)) for i in self.list_gauss]
                [print(latex(i) + " &= 0 \\\\ \\nonumber") for i in self.list_gauss[:-1]]
                print(latex(self.list_gauss[-1]) + " &= 0", "\n")

            # Solution of gauss law equations
            if self.e_op_free_input is not None:
                if self.puregauge:
                    dep_variables = list(
                        set([Symbol(j) for j in self.lattice.list_edges2_e_op])
                        - set(self.e_op_free_input)
                    )
                else:
                    dep_variables = list(
                        set(
                            [Symbol(j) for j in self.lattice.list_edges2_e_op]
                            + [Symbol(q) for q in self.q_charge_str_list]
                        )
                        - set(self.e_op_free_input)
                    )

                sol_system=solve(self.list_gauss, dep_variables, dict=True)


            else:
                sol_system=solve(self.list_gauss, dict=True)


            if len(sol_system) == 0:
                raise ValueError(
                    "The system of equations has no solution. Please check the input."
                )
            else:
                self.sol_gauss = sol_system[0]

            print("> Gauss law equations solved")

        else:
            print(">> Gauss law not applied")

        if self.gauss_law:
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
        else:
            self.e_op_free = list( set([symbols(j) for j in self.lattice.list_edges2_e_op]))

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

        if display_hamiltonian:
            print(">> Hamiltonian (symbolic + latex):")
        self._hamiltonian_el_autom()
        self._hamiltonian_mag_autom()
        self._hamiltonian_m_autom()
        self._hamiltonian_k_autom()

        self.display_hamiltonian_tot()

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
                # In case of more flavors the all contribute to the gauss law at one site
                ga_tmp = 0
                for n in range(self.n_flavors):
                    jw_node = (node[0]*self.n_flavors+n,) + node[1:]
                    ga_tmp += -1 * self.e_op_dict["q_" + self.str_node_f(jw_node)]
                    gc_tmp += self.e_op_dict["q_" + self.str_node_f(jw_node)]

            if self.static_charges_values is not None:
                if node in self.static_charges_values.keys():
                    ga_tmp -= self.static_charges_values[node]
                    if not self.puregauge:
                        gc_tmp += self.static_charges_values[node]

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

    # HAMILTONIAN
    # * symbols
    # Define Hamiltonian and apply Gauss laws

    def _hamiltonian_el_autom(self):
        """Hamiltonian for E field"""
        hamiltonian_el_sym = (Symbol(str(s)) for s in self.lattice.list_edges2_e_op)

        if self.gauss_law:
            hamiltonian_el_sym = sum(
                (
                    x**2 if x not in self.sol_gauss else (self.sol_gauss[x]) ** 2
                    for x in hamiltonian_el_sym
                )
            )  # Gauss law applied
        else:
            hamiltonian_el_sym = sum((x**2 for x in hamiltonian_el_sym))

        self.hamiltonian_el_sym = (
            hamiltonian_el_sym  # symbolic expression (useful for diplay)
        )

        if self.magnetic_basis:
            hamiltonian_el_sym = hamiltonian_el_sym.expand().subs(
                [(el**2, Symbol(str(el) + "^2")) for el in self.e_op_free]
            )
            if isinstance(self.hamiltonian_el_sym.expand(),Mul):#single operator
                self.hamiltonian_el_subs = hamiltonian_el_sym
            else:
                self.hamiltonian_el_subs = hamiltonian_el_sym.args
            print("Magnetic basis used for electric H")
        else:
            if isinstance(hamiltonian_el_sym.expand(),Mul):#single operator
                self.hamiltonian_el_subs = (
                hamiltonian_el_sym.expand()
            )  # list of symbolic expressions
            else:
                self.hamiltonian_el_subs = (
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
        # dictionary for fermionic sistes to symbols
        x_length = self.n_sites[0]
        if self.n_flavors>1:
            jw_dict = [{
                k: (
                    Symbol(f"Phi_{i+1}D", commutative=False),
                    Symbol(f"Phi_{i+1}", commutative=False),
                )
                for i, k in enumerate(self.lattice.jw_sites) if i%self.n_flavors == n
            } for n in range(self.n_flavors)]

            hamiltonian_m_sym = [[(jw_dict_flavor[i][0], jw_dict_flavor[i][1]) for i in jw_dict_flavor] for jw_dict_flavor in jw_dict]
        else:
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
                    u_op_free_dict[i][0]
                    if self.e_op_out_plus
                    else u_op_free_dict[i][
                        1
                    ]  # u_op_free_dict[i][0] if e_op_out_plus else u_op_free_dict[i][1]
                )  # if Gauss law with E out + -> U / else U^dag
            else:
                hamilt_k_elem = 1
            # phase in H_k in y-direction as Kogut Susskind H #TODO:assume 2 components spinor >check with 4 components

            for n in range(self.n_flavors):
                if i[0][1]%2 == 0:
                    start_tuple = (i[0][0]*self.n_flavors+n,) + i[0][1:]
                else:
                    start_tuple = ((i[0][0]+1)*self.n_flavors-n-1,) + i[0][1:]
                if i[1][1]%2 == 0:
                    end_tuple = (i[1][0]*self.n_flavors+n,) + i[1][1:] 
                else: 
                    end_tuple = ((i[1][0]+1)*self.n_flavors-n-1,) + i[1][1:] 
                jw_link = (start_tuple, end_tuple)
                
                if self.lattice.dims == 1:
                    phase = 1
                    hamiltonian_k_sym.append(
                        (phase, jw_dict[jw_link[0]][0], hamilt_k_elem, jw_dict[jw_link[1]][1])
                    )

                elif self.lattice.dims == 2:
                    phase = (
                        (-1) ** ((sum(i[0])) % 2) if i[0][1] != i[1][1] else 1
                    )  # change in y direction if n odd

                    xy_term = (
                        "y" if i[0][1] != i[1][1] else "x"
                    )  # if x - adjoint, if y + adjoint

                    hamiltonian_k_sym.append(
                        (xy_term, phase, jw_dict[jw_link[0]][0], hamilt_k_elem, jw_dict[jw_link[1]][1])
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
                        (i_term, phase, jw_dict[jw_link[0]][0], hamilt_k_elem, jw_dict[jw_link[1]][1])
                    )  # phi^dag U phi

                else:
                    raise ValueError("Only 1, 2 and 3 dimensions are supported.")

        self.hamiltonian_k_sym = hamiltonian_k_sym

    # display
    def display_hamiltonian_tot(self):
        """Display the total Hamiltonian of the system."""
        # ************************************  H_E   ************************************
        if self.len_e_op > 0 and self.display_hamiltonian:  # Hamiltonian to print
            # H_E only for display##############

            E_mag_subs = {
                el_eop: Sum(
                    Symbol("fs_nu")
                    * (
                        Symbol("U_" + el_eop.name[2:]) ** nu
                        - Symbol("U_" + el_eop.name[2:] + "D") ** nu
                    )
                    / (2j),
                    (nu, 1, 2 * self.ll_par),
                )
                for el_eop in self.e_op_free
            }
            # dict for substitution of E^2 operators to expression of U and U^dag for magnetic basis
            Epow2_mag_subs = {
                el_eop
                ** 2: Sum(
                    Symbol("fc_nu")
                    * (
                        Symbol("U_" + el_eop.name[2:]) ** nu
                        + Symbol("U_" + el_eop.name[2:] + "D") ** nu
                    )
                    / 2,
                    (nu, 1, 2 * self.ll_par),
                )
                + Symbol("L")
                for el_eop in self.e_op_free
            }
            hamilt_el_expand = expand(self.hamiltonian_el_sym)

            hamiltonian_el_sym_mbasis_disp = (
                (hamilt_el_expand.subs(Epow2_mag_subs))
                .subs(E_mag_subs)
                .subs(
                    Symbol("L"),
                    Symbol("id") * float(self.ll_par * (self.ll_par + 1) / 3),
                )
            )

            #############################

            h_el_embasis = (
                hamiltonian_el_sym_mbasis_disp
                if self.magnetic_basis
                else self.hamiltonian_el_sym
            )
            display_hamiltonian_el = Eq(
                Symbol("H_E"), (Symbol("g") ** 2) / 2 * h_el_embasis
            )
            display(display_hamiltonian_el)
            print(latex(display_hamiltonian_el))

        # ************************************  H_B   ************************************
        if self.len_e_op > 0:
            # Pauli expression

            if self.magnetic_basis and self.display_hamiltonian:
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

        if not self.puregauge:
            # ************************************  H_K   ************************************

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
            if self.display_hamiltonian:  # to print
                if self.n_flavors>1:
                    display_hamiltonian_m = [
                            Eq(
                                Symbol(f"H_{{m,{f+1}}}"),
                                sum(
                                    [
                                        (-1) ** j * np.prod(k)
                                        for j, k in enumerate(
                                            [
                                                (k[0].subs(k[0], Dagger(k[1])), k[1])
                                                for k in ham_m
                                            ]
                                        )
                                    ]
                                ) * Symbol(f"m_{f+1}")
                            )
                            for f, ham_m in enumerate(self.hamiltonian_m_sym)
                        ]

                    for display_hamiltonian_m_part in display_hamiltonian_m:
                        display(display_hamiltonian_m_part)
                        print(latex(display_hamiltonian_m_part))
                else:
                    display_hamiltonian_m = Eq(
                        Symbol("H_m"),
                        sum(
                            [
                                (-1) ** j * np.prod(k)
                                for j, k in enumerate(
                                    [
                                        (k[0].subs(k[0], Dagger(k[1])), k[1])
                                        for k in self.hamiltonian_m_sym
                                    ]
                                )
                            ]
                        )
                        * Symbol("m"),
                    )
                    display(display_hamiltonian_m)
                    print(latex(display_hamiltonian_m))
