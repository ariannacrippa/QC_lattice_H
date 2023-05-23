#NEW VERSION WITH ONLY HAMILTONIAN
#
#
"""Definition of the Hamiltonian for QED lattice NxN"""
from __future__ import annotations
import math
import warnings
from functools import reduce
import time
from itertools import permutations, product
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


class HamiltonianQED:

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
        nx_sites: int,
        ny_sites: int,
        g: int | float,
        fact_e_op: int | float,
        fact_b_op: int | float,
        m: int | float,
        omega: int | float,
        l: int,
        ll: int = 2,
        magnetic_basis: bool = False,
        pbc: bool = False,
        puregauge: bool = False,
        static_charges_values: dict | None = None,
        e_op_out_plus: bool = False,
        ksphase: bool = True,
        display_hamiltonian: bool = False,
        lambd:int | float = 1000.0,
        tn_comparison: bool = False,
    ) -> None:
        self.nx_sites = nx_sites
        self.ny_sites = ny_sites
        self.n_sitestot = self.nx_sites * self.ny_sites
        self.g_var = g
        self.fact_e_op = fact_e_op
        self.fact_b_op = fact_b_op
        self.m_var = m
        self.omega = omega
        self.l_par = l
        self.ll_par = ll
        self.magnetic_basis = magnetic_basis
        self.pbc = pbc
        self.puregauge = puregauge
        self.static_charges_values = static_charges_values
        self.e_op_out_plus = e_op_out_plus
        self.ksphase = ksphase
        self.display_hamiltonian = display_hamiltonian
        self.lambd = lambd
        self.tn_comparison = tn_comparison

        self.source: tuple = (0, 0)
        self.target: tuple = (
            (self.nx_sites - 1, 0)
            if self.nx_sites % 2 == 0 and self.ny_sites % 2 == 0
            else (self.nx_sites - 1, self.ny_sites - 1)
        )  # if both even or else

        self._symlist = ["I", "X", "Y", "Z", "Sd", "S-", "Su", "S+"]
        self.alpha = 2 * np.pi / (2 * self.ll_par + 1) if self.magnetic_basis else 0

        # get the start time
        start_time = time.time()
        # define graph and edges
        self.graph_lattice()
        self.graph_edges_system: list = (
            list(self._graph.edges()) + self.graph_edgesx + self.graph_edgesy
        )
        self.pos = {(x, y): (x, y) for x, y in self._graph.nodes()}

        # Build Jordan-Wigner chain, edges and plaquettes operators
        self.jw_chain_func()
        self.jw_s = [k[0] for k in self.jw_chain] + [self.target]
        self.plaquette_operators()
        self.list_edges_gauge()

        # Dictionary of all elements (E and charges q) in the system with their symbols
        self.e_op_dict = {
            s_tmp: symbols(s_tmp)
            for s_tmp in [
                "E_" + str(v_elem[0]) + str(v_elem[1]) + "x"
                if v_elem[1] == u_elem[1]
                else "E_" + str(v_elem[0]) + str(v_elem[1]) + "y"
                for v_elem, u_elem in self.graph_edges_system
            ]
            + [
                "q_" + str(x) + str(y)
                for x in range(self.nx_sites)
                for y in range(self.ny_sites)
                if self.puregauge is False
            ]
            + [
                "Q_" + str(x) + str(y)
                for x in range(self.nx_sites)
                for y in range(self.ny_sites)
                if self.puregauge is False and self.static_charges_values is not None
            ]
        }

        self.u_op_dict = {
            s_tmp: symbols(s_tmp)
            for s_tmp in [
                "U_" + str(v_elem[0]) + str(v_elem[1]) + "x"
                if v_elem[1] == u_elem[1]
                else "U_" + str(v_elem[0]) + str(v_elem[1]) + "y"
                for v_elem, u_elem in self.graph_edges_system
            ]
        }

        # #Gauss law equations in a list and display them
        self.gauss_equations()
        if self.display_hamiltonian:
            print(">> Gauss law system of equations (symbolic + latex):")
            [display(Eq(i, 0)) for i in self.list_gauss]
            [print(latex(i) + " &= 0 \\\\ \\nonumber") for i in self.list_gauss[:-1]]
            print(latex(self.list_gauss[-1]) + " &= 0", "\n")

        # Solution of gauss law equations
        self.sol_gauss = solve(self.list_gauss, dict=True)[0]

        # e_op_free from solution of Guass equations and edges
        self.e_op_free = list(
            set([symbols(j[0]) for j in self.list_edges2]).intersection(
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
                    (symbols(j[0]), symbols(k[0]))
                    for j, k in zip(self.list_edges2, self.list_edges2_u_op)
                ]
            )
            for k in self.e_op_free
        ]
        self.u_op_free_dag = [
            k.subs(
                [
                    (symbols(j[0]), Symbol(k[0] + "d"))
                    for j, k in zip(self.list_edges2, self.list_edges2_u_op)
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

        # Final result of the Hamiltonian in terms of Pauli matrices
        if self.puregauge:
            self.hamiltonian_ferm = 0
        else:
            self.hamiltonian_ferm = (
                float(self.omega) * self.hamiltonian_k_pauli
                + float(self.m_var) * self.hamiltonian_m_pauli
            )

        self.hamiltonian_gauge = (
            -self.fact_b_op / (float((self.g_var) ** 2)) * self.hamiltonian_mag_pauli
            + self.fact_e_op * float((self.g_var) ** 2) * self.hamiltonian_el_pauli
        )

        self.hamiltonian_tot = (
            self.hamiltonian_gauge + self.hamiltonian_ferm + self.hamiltonian_suppress
        ).reduce()

    def _n_qubits_g(self) -> int:
        """Returns the minimum number of qubits required with Gray encoding"""

        return int(np.ceil(np.log2(2 * self.l_par + 1)))


    def graph_lattice(self):
        """Returns the two-dimensional grid graph.
        The grid graph has each node connected to its four nearest neighbors.

        Returns
        -------
        NetworkX graph
            The (possibly periodic) grid graph of the specified dimensions.

        """

        _graph = empty_graph(0)
        rows = np.arange(self.nx_sites)
        cols = np.arange(self.ny_sites)

        _graph.add_nodes_from((i, j) for i in rows for j in cols)
        _graph.add_edges_from(
            ((i, j), (pi, j)) for pi, i in pairwise(rows) for j in cols
        )
        _graph.add_edges_from(
            ((i, j), (i, pj)) for i in rows for pj, j in pairwise(cols)
        )

        if self.pbc and self.nx_sites > 1:  # x direction
            last = rows[0]
            first = rows[-1]
            graph_edgesx = [((first, j), (last, j)) for j in cols]
        else:
            graph_edgesx = []
        if self.pbc and self.ny_sites > 1:  # y direction
            last = cols[0]
            first = cols[-1]
            graph_edgesy = [((i, first), (i, last)) for i in rows]
        else:
            graph_edgesy = []

        self._graph = _graph
        self.graph_edgesx = graph_edgesx
        self.graph_edgesy = graph_edgesy

    def draw_graph_func(self, gauss_law_fig: bool = False, savefig_dir=None):
        """Draw the graph of the lattice with the dynamical links.
        Parameters
           gauss_law_fig: bool
               If True, the free links are highlighted in gray and the dynamical
               links in black.
           savefig_dir: str
               If not None, the figure is saved in the specified directory."""

        if gauss_law_fig:
            lu_op_edges = [
                [Symbol(k[0]) for k in self.list_edges2_u_op].index(n_tmp)
                for n_tmp in self.u_op_free
            ]
            lu_op_free_map = [
                (
                    tuple(map(int, re.findall(r"\d+", self.list_edges2_u_op[i][0])[0])),
                    tuple(map(int, re.findall(r"\d+", self.list_edges2_u_op[i][1])[0])),
                )
                for i in lu_op_edges
            ]

            edge_color_list = [
                "black" if e in lu_op_free_map else "lightgray"
                for e in self.graph_edges_system
            ]
        else:
            edge_color_list = ["black" for e in self.graph_edges_system]

        fig = plt.figure(figsize=(4, 4))

        _graph = nx.DiGraph()
        for i, color_c in zip(self.graph_edges_system, edge_color_list):
            _graph.add_edge(i[0], i[1], color=color_c)

        color_map = []

        connection = "arc3, rad = 0.16" if self.pbc else "arc3, rad = 0.0"

        for node in _graph:
            if sum(node) % 2:
                color_map.append("orange")
            else:
                color_map.append("skyblue")

        # print(_graph.edges(),graph_edges_system)
        colors = nx.get_edge_attributes(_graph, "color").values()

        nx.draw(
            _graph,
            pos=self.pos,
            node_color=color_map,
            with_labels=True,
            width=2,
            edge_color=colors,
            connectionstyle=connection,
            node_size=900,
        )

        plt.axis("on")  # turns on axis
        plt.tick_params(labelleft=True, labelbottom=True)
        plt.yticks([])
        plt.xticks([])
        plt.xlabel("x", fontsize=14, fontweight="bold")
        plt.ylabel("y", fontsize=14, fontweight="bold")
        if self.pbc:
            bc_title = "PBC"
        else:
            bc_title = "OBC"

        plt.title(
            f"{self.nx_sites}x{self.ny_sites} Lattice:" + bc_title,
            fontsize=16,
            fontweight="bold",
        )

        if isinstance(savefig_dir, str):  # directory where to save figure
            plt.savefig(
                f"{savefig_dir}/system_{self.nx_sites}x{self.ny_sites}_"
                + bc_title
                + f"_gausslaw{gauss_law_fig}.pdf",
                bbox_inches="tight",
                dpi=600,
            )

        plt.show()

    # longest path in the graph #TODO: define most convenient path
    def longest_simple_paths(self) -> List[List]:
        """Returns one of the longest path between two points (source and target)
        in the graph. Advantages when Jordan-Wigner chain is defined, since we
        can avoid long expressions with a lot of global terms and long Pauli-Z strings.

        Returns
        -------
        longest_paths:list of tuples.
            List of coordinates like [(0, 0), (0, 1),..] that corresponds to one of
            the longest paths.

        """
        longest_paths = []
        longest_path_len = 0
        for path in nx.all_simple_paths(
            self._graph, source=self.source, target=self.target
        ):
            if len(path) > longest_path_len:
                longest_path_len = len(path)
                longest_paths.clear()
                longest_paths.append(path)
            elif len(path) == longest_path_len:
                longest_paths.append(path)
        return longest_paths

    # chain of elements for Jordan-Wigner function
    def jw_chain_func(self):
        """Returns one of the longest path between two points (source and target)
        in the graph. Advantages when Jordan-Wigner chain is defined, since we
        can avoid long expressions with a lot of global terms and long Z strings.


        Returns
        -------
        longest_paths:list of tuples.
            List of coordinates like [(0, 0), (0, 1),..] that corresponds to one of
            the longest paths.

        """

        jw_chain = self.longest_simple_paths()[0]
        jw_chain = [(jw_chain[i], jw_chain[(i + 1)]) for i in range(len(jw_chain) - 1)]
        not_jwchain = [
            x
            for x in self._graph.edges()
            if x not in jw_chain and x not in [(t[1], t[0]) for t in jw_chain]
        ]
        not_jwchain = list(set([tuple(sorted(t)) for t in not_jwchain]))

        if (self.n_sitestot) - (len([k[0] for k in jw_chain]) + 1) != 0:
            warnings.warn(
            "Warning: Jordan-Wigner chain has missing sites. not long enough to reach every site."
            )

        self.jw_chain = jw_chain
        self.not_jwchain = not_jwchain

    # List of edges
    def list_edges_gauge(self):
        """Returns 3 lists of edges, both for electric field E and link operator U.

        Returns
        -------

        list_edges:list
            List of edges without direction but with electric field variable E,
            used for definition of Gauss' law equations.

        list_edges2:
            List of edges with direction (x or y) and electric field variable E.


        list_edges2_u_op:
            List of edges with direction (x or y) and link variable U (for the
            definition of kinetic and magnetic term in the Hamiltonian).

        """
        list_edges = []
        list_edges2 = []
        list_edges2_u_op = []

        for v_elem, u_elem in self.graph_edges_system:
            list_edges.append(
                ("E_" + str(v_elem[0]) + str(v_elem[1]), "E_" + str(u_elem[0]) + str(u_elem[1]))
            )

            coeff = (
                "x" if v_elem[1] == u_elem[1] else "y"
            )  # x direction (y does not change)/  y direction (x does not change)

            list_edges2.append(
                (
                    "E_" + str(v_elem[0]) + str(v_elem[1]) + coeff,
                    "E_" + str(u_elem[0]) + str(u_elem[1]) + coeff,
                )
            )
            list_edges2_u_op.append(
                (
                    "U_" + str(v_elem[0]) + str(v_elem[1]) + coeff,
                    "U_" + str(u_elem[0]) + str(u_elem[1]) + coeff,
                )
            )

        self.list_edges: list = list_edges
        self.list_edges2: list = list_edges2
        self.list_edges2_u_op: list = list_edges2_u_op

    # Plaquettes operators
    def plaquette_operators(self):
        """Returns two list of plaquettes in the graph and
            in terms of link operators U.

        Returns
        -------
        plaq_list: list
                List of coordinates (tuples) for each plaquette on the lattice

        list_plaq_u_op:list
                List of strings ("U_nx(y)"), i.e. plaquettes in terms of links operator

        """

        if self.pbc is False:
            plaq_list = [
                [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
                for x in range(self.nx_sites - 1)
                for y in range(self.ny_sites - 1)
            ]
            plaq_list = [
                [(p_tmp[i % len(p_tmp)], p_tmp[(i + 1) % len(p_tmp)]) for i in range(len(p_tmp))]
                for p_tmp in plaq_list
            ]
        else:
            if self.ny_sites == 1:
                plaq_list = [
                    (i % len(range(self.nx_sites)), 0) for i in range(self.nx_sites)
                ]
                plaq_list = [
                    [
                        (
                            plaq_list[p_tmp % (len(plaq_list))],
                            plaq_list[(p_tmp + 1) % (len(plaq_list))],
                        )
                        for p_tmp in range(len(plaq_list))
                    ]
                ]
            elif self.nx_sites == 1:
                plaq_list = [
                    (0, i % len(range(self.ny_sites))) for i in range(self.ny_sites)
                ]
                plaq_list = [
                    [
                        (
                            plaq_list[p_tmp % (len(plaq_list))],
                            plaq_list[(p_tmp + 1) % (len(plaq_list))],
                        )
                        for p_tmp in range(len(plaq_list))
                    ]
                ]

            else:
                plaq_list = [
                    [
                        (x, y),
                        ((x + 1) % len(range(self.nx_sites)), y),
                        (
                            (x + 1) % len(range(self.nx_sites)),
                            (y + 1) % len(range(self.ny_sites)),
                        ),
                        (x, (y + 1) % len(range(self.ny_sites))),
                    ]
                    for x in range(self.nx_sites)
                    for y in range(self.ny_sites)
                ]

                plaq_list = [
                    [(p_tmp[i % len(p_tmp)], p_tmp[(i + 1) % len(p_tmp)])
                     for i in range(len(p_tmp))]
                    for p_tmp in plaq_list
                ]

        list_plaq_u_op = []
        for i, p_tmp in enumerate(plaq_list):
            list_p = []
            j = 0
            for v_elem, u_elem in p_tmp:
                coeff = (
                    "x" if v_elem[1] == u_elem[1] else "y"
                )  # 'x' if x direction (y does not change) / 'y' if y direction (x does not change)
                vec = u_elem if j > 1 else v_elem  # U^dag /
                list_p.append("U_" + str(vec[0]) + str(vec[1]) + coeff)
                j += 1

            list_plaq_u_op.append(list_p)

        self.plaq_list = plaq_list
        self.list_plaq_u_op = list_plaq_u_op

    # Gauss law equations in a list # TODO: generalize for non-zero total charge
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
        for i in [
            "E_" + str(x) + str(y)
            for x in range(self.nx_sites)
            for y in range(self.ny_sites)
        ]:
            coeff_0 = 0 if self.puregauge else -1

            if self.static_charges_values is not None:
                ga_tmp = coeff_0 * (symbols("q_" + i[-2:]) + symbols("Q_" + i[-2:]))
                gc_tmp += -coeff_0 * (symbols("q_" + i[-2:]) + symbols("Q_" + i[-2:]))
            else:
                ga_tmp = coeff_0 * (symbols("q_" + i[-2:]))  # charges for every sites
                gc_tmp += -coeff_0 * (symbols("q_" + i[-2:]))

            for j, k in zip(self.list_edges, self.list_edges2):
                if i in j:
                    if i == j[0]:  # E_out
                        coeff = (
                            1 if self.e_op_out_plus else -1
                        )  # if +1 then U in H_k / if -1 then U^dag in H_k
                    else:  # E_in
                        coeff = (
                            -1 if self.e_op_out_plus else 1
                        )  # if -1 then U in H_k / if 1 then U^dag in H_k

                    ga_tmp += coeff * symbols(k[0])

            list_gauss.append(ga_tmp)
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
            if isinstance(hamilt_input[0],(int, float))
            else hamilt_input[0]
        )

        for i in hamilt_input[1:]:
            if not isinstance(i, (int,float)):
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
        sgm = PauliSumOp(
            SparsePauliOp.from_sparse_list(
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
        )
        sgp = PauliSumOp(
            SparsePauliOp.from_sparse_list(
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
        )

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

    ######utilities and operators
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

    ##ENCODING FUNCTIONS AND OPERATORS
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
                for k in self.jw_s
            ]
            static_charges_subs = [
                (
                    symbols("Q_" + str(k[0]) + str(k[1])),
                    j * (I ^ (self.n_sitestot + self._n_qubits_g() * self.len_e_op)),
                )
                for k, j in zip(self.jw_s, static_charges_list)
            ]
        else:
            static_charges_subs = []

        # E FIELD
        q10 = -0.5 * (I + Z)  # JW dependent
        q00 = 0.5 * (I - Z)

        _e_op_elem = lambda i: self._e_operator(index=i + 1)
        if self.puregauge:
            e_op_field_subs = [(s_tmp, _e_op_elem(i)) for i, s_tmp in enumerate(self.e_op_free)]
            q_charges_subs = []
        else:
            e_op_field_subs = [
                (s_tmp, ((I ^ self.n_sitestot) ^ (_e_op_elem(i))))
                for i, s_tmp in enumerate(self.e_op_free)
            ]

            q_el = (
                lambda i, q: (I ^ (self.n_sitestot - 1 - i))
                ^ (q)
                ^ (I ^ (self._n_qubits_g() * self.len_e_op + i))
            )

            q_charges_subs = [
                (
                    symbols("q_" + str(k[0]) + str(k[1])),
                    q_el(i, q10),
                )
                if sum((k[0], k[1])) % 2
                else (
                    symbols("q_" + str(k[0]) + str(k[1])),
                    q_el(i, q00),
                )
                for i, k in enumerate(self.jw_s)
            ] + static_charges_subs

        # U field

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
                (s_tmp, (I ^ self.n_sitestot) ^ _u_op_elem(i))
                for i, s_tmp in enumerate(self.u_op_free)
            ] + [
                (s_tmp, ((I ^ self.n_sitestot) ^ _u_op_elem(i)).adjoint())
                for i, s_tmp in enumerate(self.u_op_free_dag)
            ]

        # ****list of Pauli substitutions for fermionic sites
        phi_el = lambda i, j: (HamiltonianQED.jw_func(i + 1, self.n_sitestot)[j]) ^ (
            I ^ (self._n_qubits_g() * self.len_u_op)
        )

        phi_jw_subs = [
            (
                Symbol(f"Phi_{i+1}d", commutative=False),
                phi_el(i, 0),
            )
            for i, k in enumerate(self.jw_s)
        ] + [
            (
                Symbol(f"Phi_{i+1}", commutative=False),
                phi_el(i, 1),
            )
            for i, k in enumerate(self.jw_s)
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
        hamiltonian_el_sym = [
            symbols("E_" + str(v_elem[0]) + str(v_elem[1]) + "x")
            if v_elem[1] == u_elem[1]
            else symbols("E_" + str(v_elem[0]) + str(v_elem[1]) + "y")
            for v_elem, u_elem in self.graph_edges_system
        ]
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
                x if symbols(x) in self.u_op_free else "id"
                for x in [k for j, k in enumerate(p_tmp)]
            ]
            for p_tmp in self.list_plaq_u_op
        ]

        # Hamiltonian for substitution
        hamiltonian_mag_subs = [
            [
                symbols(k).subs(symbols("id"), 1)
                if j < 2
                else Symbol(k + "d").subs(symbols("idd"), 1)
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
        List of tuples like [(Phi_1d, Phi_1), (Phi_2d, Phi_2),..]

        """
        hamiltonian_m_sym = []
        # dictionary for fermionic sistes to symbols

        jw_dict = {
            k: (
                Symbol(f"Phi_{i+1}d", commutative=False),
                Symbol(f"Phi_{i+1}", commutative=False),
            )
            for i, k in enumerate(self.jw_s)
        }

        for i in jw_dict:
            hamiltonian_m_sym.append((jw_dict[i][0], jw_dict[i][1]))

        self.hamiltonian_m_sym = hamiltonian_m_sym

    def _hamiltonian_k_autom(self):
        """Hamiltonian for kinetic term of the type 'phi^dag U phi'."""

        # dictionary for dynamical links to symbols
        lu_op_edges = [
            [Symbol(k[0]) for k in self.list_edges2_u_op].index(n_tmp)
            for n_tmp in self.u_op_free
        ]
        u_op_free_edges = [
            (
                tuple(map(int, re.findall(r"\d+", self.list_edges2_u_op[i][0])[0])),
                tuple(map(int, re.findall(r"\d+", self.list_edges2_u_op[i][1])[0])),
                u_elem,
                udag,
            )
            for i, u_elem, udag in zip(lu_op_edges, self.u_op_free, self.u_op_free_dag)
        ]
        u_op_free_dict = {(k[0], k[1]): (k[2], k[3]) for k in u_op_free_edges}

        # dictionary for fermionic sistes to symbols
        jw_dict = {
            k: (
                Symbol(f"Phi_{i+1}d", commutative=False),
                Symbol(f"Phi_{i+1}", commutative=False),
            )
            for i, k in enumerate(self.jw_s)
        }

        # Build Hamiltonian
        hamiltonian_k_sym = []
        for i in self.graph_edges_system:  # for every edge
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
            # phase in H_k in y-direction as Kogut Susskind H

            if self.nx_sites==1 or self.ny_sites==1:
                phase = 1
            else:
                phase = (
                    (-1) ** (sum(i[0]) % 2) if self.ksphase and i[0][1] != i[1][1] else 1
                )  # change in y direction

            hamiltonian_k_sym.append((phase, jw_dict[i[0]][0], hamilt_k_elem, jw_dict[i[1]][1]))

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
                I ^ (self.n_sitestot + self._n_qubits_g() * (self.len_u_op))
            )

        # ************************************  H_B   ************************************
        if len(self.u_op_free) > 0:
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
                                ).subs(Symbol("id", commutative=False), 1)
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
                I ^ (self.n_sitestot + self._n_qubits_g() * (self.len_u_op))
            )
        # ************************************  H_K   ************************************

        # Pauli expression
        hamiltonian_k_1 = sum(
            [
                HamiltonianQED._subs_hamilt_sym_to_pauli(h, self.u_op_field_subs + self.phi_jw_subs)
                for h in self.hamiltonian_k_sym
            ]
        )
        hamiltonian_k_pauli = (
            0.5 * (hamiltonian_k_1 + hamiltonian_k_1.adjoint())
        ).reduce()  # (must be then multiplied by omega)
        if self.display_hamiltonian:
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
                (Symbol("Omega") / 2)
                * (
                    sum(
                        [
                            Mul(*k, evaluate=False) if k[2] != 1 else Mul(*k)
                            for k in hamiltonian_k_display
                        ]
                    )
                    + Symbol("h.c.", commutative=False)
                ),
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

            suppr1 = self.lambd * (h_s)
            hamiltonian_gauge_suppr = 0.0 * (I ^ (self._n_qubits_g() * (self.len_u_op)))

            for i in range(1, self.len_u_op + 1):
                hamiltonian_gauge_suppr += (
                    (I ^ (self._n_qubits_g() * (self.len_u_op - i)))
                    ^ (suppr1)
                    ^ (I ^ (self._n_qubits_g() * (i - 1)))
                ).reduce()

        else:
            hamiltonian_gauge_suppr = 0.0 * (I ^ self.n_sitestot)

        # ****** fermion
        suppr_f = I ^ (self.n_sitestot)
        # the state is projected onto zero-charge state (fermions), same number of 1 and 0
        for i in range(2**self.n_sitestot):
            bincount = sum([1 for el in bin(i)[2:] if el == "1"])
            if bincount == self.n_sitestot / 2:
                binc = format(i, "0%db" % self.n_sitestot)
                suppr_f += -1.0 * reduce(
                    lambda x, y: (x) ^ (y), [s_down if x == "0" else s_up for x in binc]
                )

        hamiltonian_nzcharge_suppr = self.lambd * (
            (suppr_f) ^ (I ^ (self._n_qubits_g() * self.len_u_op))
        )

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
                            "0" * (self.n_sitestot // 2) + "1" * (self.n_sitestot // 2)
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

            hamiltonian_gauss_suppr = self.lambd * (suppr_gaus)

        elif self.puregauge:
            hamiltonian_gauss_suppr = 0.0 * (I ^ (self._n_qubits_g() * (self.len_u_op)))
        else:
            hamiltonian_gauss_suppr = 0.0 * (
                (I ^ self.n_sitestot) ^ (I ^ (self._n_qubits_g() * (self.len_u_op)))
            )

        if self.puregauge:
            hamiltonian_suppress = (
                (hamiltonian_gauge_suppr) + (hamiltonian_gauss_suppr)
            ).reduce()
        elif self.len_u_op > 0:
            hamiltonian_suppress = (
                ((I ^ self.n_sitestot) ^ hamiltonian_gauge_suppr)
                + (hamiltonian_nzcharge_suppr)
                + (hamiltonian_gauss_suppr)
            ).reduce()
        else:  # no gauge fields
            hamiltonian_suppress = ((hamiltonian_nzcharge_suppr)).reduce()

        self.hamiltonian_suppress = hamiltonian_suppress



