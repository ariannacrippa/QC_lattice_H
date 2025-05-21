#
# Copyright (C) 2004-2018, NetworkX Developers
# Aric Hagberg <hagberg@lanl.gov>
# Dan Schult <dschult@colgate.edu>
# Pieter Swart <swart@lanl.gov>
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.

#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.

#   * Neither the name of the NetworkX Developers nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Definition of the Hamiltonian for QED lattice NxN"""
from __future__ import annotations
import warnings
from itertools import repeat, combinations
import re
import numpy as np
import networkx as nx
from networkx import (
    cycle_graph,
    path_graph,
    relabel_nodes,
    get_edge_attributes,
)
from networkx.algorithms.operators.product import cartesian_product
from networkx.generators.classic import empty_graph
from networkx.utils import flatten
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
from sympy import Symbol

mpl.rcParams["patch.facecolor"] = "xkcd:white"


class HCLattice:

    """The algorithm generates a generic N dimensional hypercubic lattice
    with open or periodic boundary conditions.

    Parameters
    ----------

    n_sites: list
            Number of sites in the lattice in the nth direction.

    pbc : bool
            If `pbc` is True, both dimensions are periodic. If False, none
            are periodic.

    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        n_sites: list,
        pbc: bool | list = False,
        puregauge: bool = True,
        n_flavors: int = 1,
    ) -> None:
        self.n_sites = n_sites
        while 1 in self.n_sites:  # avoid 1 dimension useless input
            self.n_sites.remove(1)
        self.n_sitestot = np.prod(self.n_sites)
        self.dims = len(self.n_sites)  # how many dimensions
        self.pbc = pbc
        self.puregauge = puregauge
        self.n_flavors = n_flavors  # number of flavors, default 1 

        # define graph and edges
        self.graph_lattice()
        self.graph_edges_system: list = list(self.graph.edges)
        # Build  edges and plaquettes operators
        self.list_edges_gauge()
        self.plaquette_operators()
        self.total_links()
        # Build Jordan-Wigner chain
        self.jw_chain_func()
        self.func_qstatic_dist()
        # self.find_gausslaw_links()

    # ADDED: create_using=nx.DiGraph and periodic inverted
    def graph_lattice(self):
        """Returns the *n*-dimensional grid graph.

        The dimension *n* is the length of the list `dim` and the size in
        each dimension is the value of the corresponding list element.

        Parameters
        ----------
        dim : list or tuple of numbers or iterables of nodes
            'dim' is a tuple or list with, for each dimension, either a number
            that is the size of that dimension or an iterable of nodes for
            that dimension. The dimension of the grid_graph is the length
            of `dim`.

        periodic : bool or iterable
            If `periodic` is True, all dimensions are periodic. If False all
            dimensions are not periodic. If `periodic` is iterable, it should
            yield `dim` bool values each of which indicates whether the
            corresponding axis is periodic.

        Returns
        -------
        NetworkX graph
            The (possibly periodic) grid graph of the specified dimensions.

        """

        if not self.n_sites:
            return empty_graph(0)

        try:
            func = (cycle_graph if p else path_graph for p in self.pbc)
        except TypeError:
            func = repeat(cycle_graph if self.pbc else path_graph)

        graph_g = next(func)(self.n_sites[0], create_using=nx.DiGraph)
        for current_dim in self.n_sites[1:]:
            _graph_new = next(func)(current_dim, create_using=nx.DiGraph)
            graph_g = cartesian_product(graph_g, _graph_new)
        graph = relabel_nodes(graph_g, flatten)

        self.graph = graph

    def draw_graph_func(
        self,
        gauss_law_fig: bool = False,
        e_op_free=None,
        static_charges=None,
        savefig_dir=None,
        figname=None,
        suptitle=False,
        weight: dict | None = None,
    ):
        """Draw the graph of the lattice with the dynamical links.
        Parameters
            gauss_law_fig: bool
               If True, the free links are highlighted in gray and the dynamical
               links in black.

            e_op_free: list
                List of dynamical links after Gauss law is applied.

            savefig_dir: str
               If not None, the figure is saved in the specified directory.

            figname: str
                If not None, the figure is saved with this name. Otherwise, a default name is used.

            suptitle: bool
                If True, the title of the figure is displayed.

            weight: dict
                Dictionary of the form {E_00x:1} with E_00x the electric field
                variable and 1 the value of the electric field.
        """
        # edges
        if gauss_law_fig and e_op_free is not None:
            lu_op_edges = [
                [Symbol(k) for k in self.list_edges2_e_op].index(n_tmp)
                for n_tmp in e_op_free
            ]
            to_int = (
                lambda i, cff: tuple(
                    map(int, re.findall(r"\d+", self.list_edges[i][cff])[0])
                )[0]
                if self.dims == 1
                else tuple(map(int, re.findall(r"\d+", self.list_edges[i][cff])[0]))
            )
            lu_op_free_map = [
                (
                    to_int(i, 0),
                    to_int(i, 1),
                )
                for i in lu_op_edges
            ]  # list of edges (tuples) that are dynamical

            # edge_color_list = [
            #     "black" if e in lu_op_free_map else "lightgray"
            #     for e in self.graph_edges_system
            # ]

            edge_color_list = ["black" for e in self.graph_edges_system]
            edges_linestyle = [
                "solid" if e in lu_op_free_map else (0, (5, 10))
                for e in self.graph_edges_system
            ]

        elif not gauss_law_fig and e_op_free is not None:
            raise ValueError("gauss_law_fig must be True if e_op_free is not None")
        elif gauss_law_fig and e_op_free is None:
            raise ValueError("e_op_free must be not None if gauss_law_fig is True")
        else:
            edge_color_list = ["black" for e in self.graph_edges_system]
            edges_linestyle = ["solid" for e in self.graph_edges_system]

        if weight:
            dict_label = {}
            for (
                key,
                val,
            ) in (
                weight.items()
            ):  # weight is dict of type E_00x:1 with val the values of el. field
                key = key.name

                # List of directions for the edges if D<=3 or more
                if self.dims < 4:
                    ax_direct = ["x", "y", "z"]
                else:
                    ax_direct = list(map(chr, range(97, 123)))
                if key[0] == "E":
                    inpoint = [int(n) for n in re.findall(r"\d+", key)[0]]
                    # inpoint2=inpoint.copy()
                    # direction = ax_direct.index(key[-1])
                    # inpoint2[direction]+=1%n_sites[direction]
                    # inpoint2[direction]=inpoint2[direction]%n_sites[direction]
                    inpoint2 = [
                        (inpoint[i] + (1 if i == ax_direct.index(key[-1]) else 0))
                        % self.n_sites[i]
                        for i in range(len(inpoint))
                    ]

                    dict_label[tuple([tuple(inpoint), tuple(inpoint2)])] = val
            self.dict_label = dict_label

        fig = plt.figure(figsize=(8, 6))

        color_map = []

        # nodes
        for node in self.graph:
            if (
                static_charges is not None
            ):  # TODO: again condition of negative static charges for even sites/positive for odd sites
                col = (
                    "cornflowerblue"
                    if node in static_charges.keys() and static_charges[node] < 0
                    else "red"
                    if node in static_charges.keys() and static_charges[node] > 0
                    else "lightgray"
                )
                color_map.append(col)
            else:
                if self.dims == 1:
                    if node % 2:
                        color_map.append("orange")
                    else:
                        color_map.append("skyblue")
                else:
                    if sum(node) % 2:
                        color_map.append("orange")
                    else:
                        color_map.append("skyblue")

        connection = "arc3, rad = 0.16" if self.pbc else "arc3, rad = 0.0"

        if self.dims == 1 or self.dims > 3:
            for i, color_c in zip(self.graph_edges_system, edge_color_list):
                self.graph.add_edge(i[0], i[1], color=color_c)
            colors = nx.get_edge_attributes(self.graph, "color").values()

            nx.draw(
                self.graph,
                pos=nx.circular_layout(
                    self.graph
                ),  # TODO find best layout for 1D or D>3
                node_color=color_map,
                with_labels=True,
                width=2,
                edge_color=colors,
                connectionstyle=connection,
                node_size=600 * self.dims,
            )

        else:
            ax_plt = fig.add_subplot(111, projection="3d")
            ax_plt.scatter(
                *np.array(self.graph.nodes).T, s=300, c=color_map
            )  # ,alpha=1)

            # Nodes labels
            q_label_dict = {}
            if weight:
                q_label_dict = {
                    str([int(n) for n in re.findall(r"\d+", key.name)[0]]): str(val)
                    for key, val in weight.items()
                    if key.name[0] == "q"
                }

            if len(q_label_dict) > 0:
                for nds, val in q_label_dict.items():
                    if self.dims == 2:
                        ax_plt.text(*np.array(eval(nds)) - 0.03, 0, val, fontsize=14)
                    else:
                        ax_plt.text(*np.array(eval(nds)) - 0.03, val, fontsize=14)

            else:
                for nds in np.array(self.graph.nodes):
                    if self.dims == 2:
                        ax_plt.text(
                            *nds - 0.02,
                            0,
                            "(" + ",".join(map(str, nds)) + ")",
                            fontsize=10,
                        )
                    else:
                        ax_plt.text(
                            *nds - 0.02,
                            "(" + ",".join(map(str, nds)) + ")",
                            fontsize=10,
                        )

            # Plot the edges
            arrow_options = lambda linestyle: dict(
                arrowstyle="-|>",
                mutation_scale=15,
                lw=1.5,
                connectionstyle=connection,
                linestyle=linestyle,
            )

            # label the edges if weight
            if weight:
                label_edges = [str(self.dict_label[i]) for i in self.graph_edges_system]
            else:
                label_edges = [None for i in self.graph_edges_system]

            # for vizedge, col,lbl in zip(np.array(self.graph.edges), edge_color_list,label_edges):

            for vizedge, col, lbl, linestyle in zip(
                np.array(self.graph.edges),
                edge_color_list,
                label_edges,
                edges_linestyle,
            ):
                if weight:
                    if int(lbl) > 0:
                        clr_lbl = "red"
                    elif int(lbl) < 0:
                        clr_lbl = "cornflowerblue"
                    else:
                        clr_lbl = "grey"
                    if self.dims == 2:
                        arrow = Arrow3D(
                            *vizedge.T,
                            [0, 0],
                            ec=clr_lbl,
                            color=clr_lbl,
                            linewidth=5,
                            **arrow_options(linestyle),
                        )
                    else:
                        arrow = Arrow3D(
                            *vizedge.T,
                            ec=clr_lbl,
                            color=clr_lbl,
                            **arrow_options(linestyle),
                        )

                    # Add label the edges if weight
                    ax_plt.annotate(
                        lbl,
                        (0.5, 0.5),
                        xycoords=arrow,
                        ha="center",
                        va="bottom",
                        color=clr_lbl,
                        fontsize=14,
                    )
                else:
                    if self.dims == 2:
                        arrow = Arrow3D(
                            *vizedge.T,
                            [0, 0],
                            ec=col,
                            color=col,
                            linewidth=5,
                            **arrow_options(linestyle),
                        )
                    else:
                        arrow = Arrow3D(
                            *vizedge.T, ec=col, color=col, **arrow_options(linestyle)
                        )

                ax_plt.add_artist(arrow)

            def _format_axes(ax_plt):
                """Visualization options for the 3D axes."""
                for dim in (ax_plt.xaxis, ax_plt.yaxis, ax_plt.zaxis):
                    dim.set_ticks([])
                ax_plt.set_xlabel("x")
                ax_plt.set_ylabel("y")
                ax_plt.set_zlabel("z")

            _format_axes(ax_plt)


        if self.pbc:
            bc_title = "PBC"
        else:
            bc_title = "OBC"
        if suptitle:

            fig.patch.set_facecolor("xkcd:white")

            fig.suptitle(
                "x".join(map(str, self.n_sites)) + f" Lattice:" + bc_title,
                fontsize=12,
                fontweight="bold",
                horizontalalignment="center",
                color="black",
            )

        # Add the colorbar
        if static_charges is not None:
            import matplotlib.lines as mlines

            handles = []
            for key, val in static_charges.items():
                if val > 0:  # charge pos
                    qstat_o = val
                    red_c = mlines.Line2D(
                        [],
                        [],
                        color="red",
                        marker="o",
                        linestyle="None",
                        markersize=10,
                        label=f"Q={qstat_o}",
                    )
                    handles.append(red_c)
                else:  # charge neg
                    qstat_e = val
                    blue_c = mlines.Line2D(
                        [],
                        [],
                        color="cornflowerblue",
                        marker="o",
                        linestyle="None",
                        markersize=10,
                        label=f"Q={qstat_e}",
                    )
                    handles.append(blue_c)

            grey_c = mlines.Line2D(
                [],
                [],
                color="lightgray",
                marker="o",
                linestyle="None",
                markersize=10,
                label="Q=0",
            )
            radius = mlines.Line2D(
                [],
                [],
                color="black",
                marker="_",
                linestyle="None",
                markersize=10,
                label=f"r={np.round(self.distance_f(*static_charges.keys()),3)}",
            )
            plt.legend(
                handles=handles + [grey_c, radius],
                loc="upper right",
                bbox_to_anchor=(1.1, 1.1),
            )

        if isinstance(savefig_dir, str):  # directory where to save figure
            if isinstance(figname, str):
                namefig = figname
            else:
                namefig = (
                    "system_"
                    + "x".join(map(str, self.n_sites))
                    + "_"
                    + bc_title
                    + f"_gausslaw{gauss_law_fig}"
                )
                if static_charges:
                    namefig += f"_staticcharges{static_charges}"
                namefig += ".png"

            fig.savefig(
                f"{savefig_dir}/" + namefig,
                bbox_inches="tight",
                dpi=600,
            )
        plt.show()

    # # List of edges
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
        list_edges2_e_op = []
        list_edges2_u_op = []

        # List of directions for the edges if D<=3 or more
        if self.dims < 4:
            ax_direct = ["x", "y", "z"]
        else:
            ax_direct = list(map(chr, range(97, 123)))

        for (
            u_elem,
            v_elem,
        ) in self.graph_edges_system:  # TODO check d in dagger U_d later
            op_chr_u = "U_"
            op_chr_e = "E_"
            coeff = "none"

            if self.dims == 1:
                edge_op1, edge_op2 = str(u_elem), str(v_elem)
                coeff = ax_direct[0]  # Find in which direction the edge is

            else:
                edge_op1, edge_op2 = "".join(map(str, u_elem)), "".join(
                    map(str, v_elem)
                )
                # Find in which direction the edge is
                for index, (first, second) in enumerate(zip(u_elem, v_elem)):
                    if first != second:
                        coeff = ax_direct[index]

            # list of edges starting point and ending point
            list_edges.append((op_chr_e + edge_op1, op_chr_e + edge_op2))

            # list of edges with direction #TODO if okay, remove the list_edges2_e/u_op below
            list_edges2_e_op.append(op_chr_e + edge_op1 + coeff)
            list_edges2_u_op.append(op_chr_u + edge_op1 + coeff)

            # list_edges2_e_op.append(
            #     (op_chr_e + edge_op1 + coeff, op_chr_e + edge_op2 + coeff)
            # )
            # list_edges2_u_op.append(
            #     (op_chr_u + edge_op1 + coeff, op_chr_u + edge_op2 + coeff)
            # )

        self.list_edges: list = list_edges
        self.list_edges2_e_op: list = list_edges2_e_op
        self.list_edges2_u_op: list = list_edges2_u_op

    # # Plaquettes operators
    def plaquette_operators(self):
        """Returns two list of plaquettes in the graph and
            in terms of link operators U.

        Returns
        -------
        plaq_list: list
                List of coordinates (tuples) for each plaquette on the lattice

        list_plaq_u_op:list
                List of strings ("U_n(dir.)"), i.e. plaquettes in terms of links operator
                with n the site and dir. the direction of the edge.

        """

        # TODO: test for dims>4 how to visualize it?
        plaq_list = []

        if 1 < self.dims < 4:  # 2D or 3D
            for a_index, b_index in list(
                combinations(range(self.dims), 2)
            ):  # change only 2 coordinates
                list_tmp = []
                for node in self.graph.nodes:
                    if isinstance(
                        self.pbc, list
                    ):  # check which directions are self. if self. iterable
                        op_tmp = (
                            lambda n, j: (n + 1) % self.n_sites[j]
                            if self.pbc[j]
                            else n + 1
                        )
                    elif self.pbc is True:  # all directions pbc if pbc True
                        op_tmp = lambda n, j: (n + 1) % self.n_sites[j]
                    else:  # no pbc
                        op_tmp = lambda n, j: n + 1

                    tpl1 = node

                    tpl2 = tuple(
                        op_tmp(n, j) if j == a_index else n for j, n in enumerate(node)
                    )
                    if tpl2 not in self.graph.nodes:
                        continue

                    tpl3 = tuple(
                        op_tmp(n, i) if i in (a_index, b_index) else n
                        for i, n in enumerate(node)
                    )
                    if tpl3 not in self.graph.nodes:
                        continue

                    tpl4 = tuple(
                        op_tmp(n, j) if j == b_index else n for j, n in enumerate(node)
                    )
                    if tpl4 not in self.graph.nodes:
                        continue

                    list_tmp.append([tpl1, tpl2, tpl3, tpl4])

                plaq_list.append(list_tmp)

            plaq_list = [item for sublist in plaq_list for item in sublist]

            plaq_list = [
                [
                    (p_tmp[i % len(p_tmp)], p_tmp[(i + 1) % len(p_tmp)])
                    for i in range(len(p_tmp))
                ]
                for p_tmp in plaq_list
            ]

        if self.dims < 4:
            ax_direct = ["x", "y", "z"]
        else:
            ax_direct = list(map(chr, range(97, 123)))

        list_plaq_u_op = []
        for i, p_tmp in enumerate(plaq_list):
            list_p = []
            j = 0
            for v_elem, u_elem in p_tmp:
                op_chr_u = "U_"
                coeff = "none"

                for index, (first, second) in enumerate(zip(u_elem, v_elem)):
                    if first != second:
                        coeff = ax_direct[index]

                vec = u_elem if j > 1 else v_elem  # U^dag after 2 edges: UUU^dagU^dag

                list_p.append(op_chr_u + "".join(map(str, vec)) + coeff)
                j += 1

            list_plaq_u_op.append(list_p)

        self.plaq_list = plaq_list
        self.list_plaq_u_op = list_plaq_u_op


    def total_links(self):
        """Compute number of dynamical links before/after Gauss's law
        Input:
        n_sites: list of number of sites in each dimension
        pbc: boolean, if True periodic boundary conditions are applied
        Output:
        links_before_g: number of links
        links_after_g: number of dynamical links after Gauss's law constraints
        """
        if self.pbc:
            links_before_g = int(self.n_sitestot*len(self.n_sites))
        else:
            links_before_g = int(round(self.n_sitestot * sum((ni - 1) / ni for ni in self.n_sites)))

        #Gauss law
        links_after_g= int(round(links_before_g - (self.n_sitestot - 1)))

        self.links_before_g = links_before_g
        self.links_after_g = links_after_g

    def find_gausslaw_links(self,print_res=False):

        """
        Find the links that are used in the Gauss law operator.
        This function identifies the links that are adjacent to two plaquettes
        and are not already selected.
        It returns a list of selected links.

        Input:
        print_res: bool
            If True, the function prints the number of selected links for each plaquette.

        """

        #TODO extend to 3d

        if len(self.n_sites) !=2:
            raise ValueError("This function is only implemented for 2D lattices.")


        def build_link_to_plaquettes(plaquettes):
            """ Function to build a dictionary with the links and the plaquettes they are adjacent to.
            Input:
            - plaquettes: list of plaquettes
            Output:
            - link_to_plaquettes: dictionary with the links and the plaquettes they are adjacent to
            - plaq_to_index: dictionary with the plaquettes as keys and their indices as values
            """
            link_to_plaquettes = {}
            plaq_to_index = {}
            for i, plaq in enumerate(plaquettes):
                plaq_to_index[str(plaq)] = i
                for link in plaq:
                    link_to_plaquettes.setdefault(link, set()).add(i)
            return link_to_plaquettes, plaq_to_index

        def plaq_count(list_plaq_u_op,selected_links):
            """ Function to count how many selected links are in every plaquette."""
            plaq_tot_count={}
            for plaq in list_plaq_u_op:#count how many selected links are in every plaq
                count=0
                if len(selected_links)==0:
                    count=0
                else:
                    for link in selected_links:
                        if link in plaq:
                            count+=1
                plaq_tot_count[str(plaq)]=count
            return plaq_tot_count


        def build_selected_links(link_to_plaquettes):
            """Function to select the links that are used in the Gauss law operator.
            This function identifies the links that are adjacent to two plaquettes
            and are not already selected.
            It returns a list of selected links.
            Input:
            - link_to_plaquettes: dictionary with the links and the plaquettes they are adjacent to
            - selected_links: list of selected links
            - used_plaq: list of used plaquettes
            Output:
            - selected_links: list of selected links
            """
            if self.n_sites[0] > 10 or self.n_sites[1] > 10:#TODO: valid for lattice <11x11!U_nxny must be converted to Unx_ny if nx,ny has more that 1 digit
                raise ValueError("Lattice size too large. Please use a smaller lattice size.")

            used_plaq = []
            used_plaq2 = []
            selected_links = []

            min_flag = 0
            for link, plaquettes in link_to_plaquettes.items():#first iteration for links only one plaquette if any
                if link not in selected_links and len(plaquettes) == 1 and next(iter(plaquettes)) not in used_plaq:  # if only one plaquette is used
                    selected_links.append(link)
                    used_plaq.extend(plaquettes)
                    if len(selected_links) == self.links_after_g:
                        return selected_links


            itr=0
            while len(selected_links) < self.links_after_g:
                itr += 1
                if itr > 1000:
                    raise ValueError("Too many iterations.")
                    break

                recompute = False  # Flag to indicate restart of the while loop

                plaq_tot_count = plaq_count(self.list_plaq_u_op, selected_links)

                min_count = min(plaq_tot_count.values())  # Find the minimum value
                min_plaq_indices = [plaq_idx for plaq_idx, count in plaq_tot_count.items() if count == min_count]

                if min_count != min_flag:#Function to count minimum value, if changes, reset used_plaq2
                    min_flag = min_count
                    used_plaq2 = []

                for plaq_idx in min_plaq_indices:
                    plaq = eval(plaq_idx)  # Convert string back to list
                    for link in plaq:
                        if link not in selected_links:
                            if all(plaquette not in used_plaq for plaquette in link_to_plaquettes[link]):

                                plaquettes = link_to_plaquettes[link]
                                selected_links.append(link)
                                if len(selected_links) == self.links_after_g:
                                    return selected_links
                                used_plaq.extend(plaquettes)
                                recompute = True  # Set flag to recompute from start of while
                                break  # Break out of inner for loop

                    if recompute:
                        break  # Break out of outer for loop
                    else:
                        if min_count == 0:  # Check for plaquettes with zero selected links
                            for link in plaq:
                                if link not in selected_links:

                                    plaquettes = link_to_plaquettes[link]
                                    selected_links.append(link)
                                    if len(selected_links) == self.links_after_g:
                                        return selected_links
                                    used_plaq.extend(plaquettes)
                                    recompute = True
                                    break  # Break out of inner for loop
                        for link in plaq:
                            if link not in selected_links and len(link_to_plaquettes[link]) == 1: # If only one plaquette is used

                                plaquettes = link_to_plaquettes[link]
                                selected_links.append(link)
                                if len(selected_links) == self.links_after_g:
                                    return selected_links
                                used_plaq2.extend(plaquettes)

                                recompute = True
                                break  # Break out of inner for loop
                        if recompute:
                            break
                        for link in plaq:
                            if link not in selected_links and all(plaquette not in used_plaq2 for plaquette in link_to_plaquettes[link]):


                                plaquettes = link_to_plaquettes[link]
                                selected_links.append(link)
                                if len(selected_links) == self.links_after_g:
                                    return selected_links
                                used_plaq2.extend(plaquettes)


                                recompute = True
                                break

                        if recompute:
                            break
                    if recompute:
                        break # Break out of outer for loop

                if recompute:
                    continue  # Start from beginning of while loop
                elif self.pbc:
                    for plaq_idx in min_plaq_indices:
                        plaq = eval(plaq_idx)

                        for link in plaq:
                            if link not in selected_links:
                                plaquettes = link_to_plaquettes[link]
                                selected_links.append(link)
                                if len(selected_links) == self.links_after_g:
                                    return selected_links
                                used_plaq2.extend(plaquettes)

                                recompute = True
                                break
                    if recompute:
                        continue

            return selected_links



        link_to_plaquettes, plaq_to_index = build_link_to_plaquettes(self.list_plaq_u_op)

        # Keep track of already used plaquettes
        selected_links=build_selected_links(link_to_plaquettes)


        plaq_tot_count=plaq_count(self.list_plaq_u_op,selected_links)

        if print_res:
            if len(selected_links)< self.links_after_g:
                print('selected links:',len(selected_links),'lower than expected:',self.links_after_g)
            else:
                print('OK! selected links:',len(selected_links),'expected:',self.links_after_g)

            print('Link before gauss =',self.links_before_g,
            '\nLink after gauss =',self.links_after_g,
            '\nLink selection =',len(selected_links),
            '\nN.er of selected links for each plaquette:\n',plaq_tot_count.values())


        selected_links=[Symbol('E'+link[1:]) for link in selected_links]


        self.selected_links=selected_links
        self.plaq_tot_count=plaq_tot_count



    # # build the jw chain until 3D #TODO how to do this for D>3?
    # JW chain doesn't matter if pbc or not
    def jw_chain_func(self):
        """Returns a chain that connects all the sites in a single path.
        in the graph. Advantages when Jordan-Wigner chain is defined, since we
        can avoid long expressions with a lot of global terms and long Z strings.

        Returns
        -------
        jw_sites: list
                List of coordinates (tuples) for each site on the lattice

        jw_chain: list
                List of edges (tuples) that connect all the sites in a single path.

        not_jw_chain: list
                List of edges (tuples) that are not in JW chain.

        """

        jw_chain = []

        if self.dims == 1:
            for x_step in range(self.n_flavors*self.n_sites[0]):
                jw_chain.append(x_step)

        elif self.dims > 1:
            range_z = (
                range(self.n_sites[2])
                if self.dims == 3
                else [
                    0,
                ]
            )
            for z_step in range_z:
                range_y = (
                    range(self.n_sites[1])
                    if z_step % 2 == 0
                    else range(self.n_sites[1])[::-1]
                )  # if z even, y vs up, else y vs down
                for y_step in range_y:
                    if z_step % 2 != 0:  # if z odd opposite x,y
                        range_x = (
                            range(self.n_flavors*self.n_sites[0])
                            if y_step % 2 != 0
                            else range(self.n_flavors*self.n_sites[0])[::-1]
                        )  # if y odd, x vs right, else x vs left
                    else:  # if z even regular x,y as 2D
                        range_x = (
                            range(self.n_flavors*self.n_sites[0])
                            if y_step % 2 == 0
                            else range(self.n_flavors*self.n_sites[0])[::-1]
                        )  # if y even, x vs right, else x vs left

                    for x_ch in range_x:
                        jw_chain.append(
                            (
                                x_ch,
                                y_step,
                                z_step,
                            )
                        ) if self.dims == 3 else jw_chain.append(
                            (
                                x_ch,
                                y_step,
                            )
                        )

        jw_sites = jw_chain

        jw_chain = [(jw_chain[i], jw_chain[(i + 1)]) for i in range(len(jw_chain) - 1)]
        not_jwchain = [
            x
            for x in self.graph.edges()
            if x not in jw_chain and x not in [(t[1], t[0]) for t in jw_chain]
        ]
        not_jwchain = list({tuple(sorted(t)) for t in not_jwchain})

        if (self.n_flavors*self.n_sitestot) - (len([k[0] for k in jw_chain]) + 1) != 0:
            warnings.warn(
                "Warning: Jordan-Wigner chain has missing sites. not long enough to reach every site."
            )

        self.jw_sites = jw_sites
        self.jw_chain = jw_chain
        self.not_jwchain = not_jwchain

    def distance_f(self, *points):
        if len(points) < 2:
            raise ValueError(
                "At least two points are required to calculate the distance."
            )
        if self.dims == 1:
            return np.abs(points[0] - points[1])
        else:
            return np.sqrt(sum((x - y) ** 2 for x, y in zip(points[0], points[1])))

    def func_qstatic_dist(self, charge: tuple = None, ch_val: int = None):
        """
        Input:
        charge: tuple
            Coordinates of the second charge.
        ch_val: int
            Value of the second charge.

        Returns two lists:
        1. A list of dictionaries of the form {charge:1,j:-1} where j is the coordinate of the site to which charge is connected.
        The default is the origin (0,0,...,0) and the sites to which it is connected are the odd sites.
        If the charge is on an odd site, then it is connected to the even sites.
        2. A list of distances between the charge and the sites to which it is connected.

        The choice of charges respects the Gauss law, i.e. the sum of the charges is zero and if fermions, it respect staggered formulation.
        If puregauge then opposite charges can be anywhere."""

        # set initial charge position and charge value:(fermions (staggered m>0) have charge q= 1, antifermions (staggered m<0) have charge q=-1)
        if charge is None and ch_val is None:
            charge = (0,) * self.dims
            ch_val_e, ch_val_o = -1, 1
        elif charge is None and ch_val is not None:
            charge = (0,) * self.dims
            ch_val_e, ch_val_o = (ch_val, -ch_val) if ch_val < 0 else (-ch_val, ch_val)
        elif charge is not None and ch_val is None:
            ch_val_e, ch_val_o = -1, 1
        else:
            if sum(charge) % 2 and ch_val < 0 or not sum(charge) % 2 and ch_val > 0:
                raise ValueError(
                    "Charge and charge value must be such that the charge on even(odd) site is negative(positive), since convention used is fermions(antifermions) on even(odd) sites."
                )  # TODO: check if ok. balance dynamical charge so net charge zero
            else:
                ch_val_e, ch_val_o = (
                    (ch_val, -ch_val) if ch_val < 0 else (-ch_val, ch_val)
                )

        if self.dims == 1:
            if len(charge) != 1:
                raise ValueError("Charge must be a tuple of length 1 for a 1D lattice.")

            if self.puregauge:
                distances_coord = np.array(
                    [
                        {charge[0]: ch_val_e, j: ch_val_o}
                        for j in list(self.graph.nodes)
                        if j != charge
                    ],
                    dtype=object,
                )  # puregaguge connect all sites
            else:
                if (charge[0] + 1) % 2:  # even site
                    distances_coord = np.array(
                        [
                            {charge[0]: ch_val_e, j: ch_val_o}
                            for j in list(self.graph.nodes)
                            if j % 2 and j != charge
                        ],
                        dtype=object,
                    )  # connect (0,0) to only odd sites
                else:
                    distances_coord = np.array(
                        [
                            {charge[0]: ch_val_o, j: ch_val_e}
                            for j in list(self.graph.nodes)
                            if (j + 1) % 2 and j != charge
                        ],
                        dtype=object,
                    )  # connect (odd,) to only even sites
        else:
            if len(charge) != self.dims:
                raise ValueError(
                    "Charge must be a tuple of length self.dims for a {}D lattice.".format(
                        self.dims
                    )
                )

            if self.puregauge:
                distances_coord = np.array(
                    [
                        {charge: ch_val_e, j: ch_val_o}
                        for j in list(self.graph.nodes)
                        if j != charge
                    ],
                    dtype=object,
                )
            else:
                if (sum(charge) + 1) % 2:  # even site
                    distances_coord = np.array(
                        [
                            {charge: ch_val_e, j: ch_val_o}
                            for j in list(self.graph.nodes)
                            if sum(j) % 2 and j != charge
                        ],
                        dtype=object,
                    )  # connect (0,0) to only odd sites
                else:
                    distances_coord = np.array(
                        [
                            {charge: ch_val_o, j: ch_val_e}
                            for j in list(self.graph.nodes)
                            if (sum(j) + 1) % 2 and j != charge
                        ],
                        dtype=object,
                    )

        r_list = np.empty(len(distances_coord), dtype=object)
        for i, dd in enumerate(distances_coord):
            r_list[i] = self.distance_f(*dd.keys())

        self.ch_val_e = ch_val_e
        self.ch_val_o = ch_val_o
        self.distances_coord = distances_coord
        self.r_list = r_list


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)
