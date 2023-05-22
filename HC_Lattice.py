#
#
"""Definition of the Hamiltonian for QED lattice NxN"""
from __future__ import annotations
import math
import warnings
from functools import reduce
import time
from itertools import permutations, product,repeat,combinations
import re
from typing import List
import numpy as np
import networkx as nx
from networkx import (cycle_graph,path_graph,relabel_nodes,all_simple_paths, get_edge_attributes)
from networkx.algorithms.operators.product import cartesian_product
from networkx.generators.classic import empty_graph
from networkx.utils import flatten
from qiskit.opflow import Z, X, Y, I, PauliSumOp,OperatorBase
from qiskit.quantum_info import SparsePauliOp
from IPython.display import display
from scipy import special as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
mpl.rcParams["patch.facecolor"]='xkcd:white'
from sympy import Symbol, symbols, solve, lambdify, Mul, Eq, latex
from sympy.physics.quantum.dagger import Dagger


class HCLattice:

    """The algorithm a generic N dimensional hypercubic lattice with both open and
    periodic boundary conditions.

    Parameters
    ----------

    nx_sites,ny_sites: int
            Number of sites in the lattice in the x and y direction.


    pbc : bool
            If `pbc` is True, both dimensions are periodic. If False, none
            are periodic.

    puregauge: bool
        If False, then we have fermionic degrees of freedom in the system, if True only
        gauge fields.


    """

    def __init__(
        self,
        n_sites: list ,
        pbc: bool | list= False,
        puregauge: bool = False,
    ) -> None:
        self.n_sites = n_sites
        self.n_sitestot = np.prod(self.n_sites)
        while 1 in self.n_sites:#avoid 1 dimension useless input
            self.n_sites.remove(1)

        self.dims = len(self.n_sites) #how many dimensions
        self.pbc = pbc
        self.puregauge = puregauge

        self.nx_sites = self.n_sites[0]
        self.ny_sites = 1#self.n_sites[1]

        self.source: tuple = (0, 0)
        self.target: tuple = (
            (self.nx_sites - 1, 0)
            if self.nx_sites % 2 == 0 and self.ny_sites % 2 == 0
            else (self.nx_sites - 1, self.ny_sites - 1)
        )  # if both even or else

        # define graph and edges
        self.graph_lattice()
        self.graph_edges_system: list = (
            list(self._graph.edges)
        )

        # Build Jordan-Wigner chain, edges and plaquettes operators
        # self.jw_chain_func()
        # self.jw_s = [k[0] for k in self.jw_chain] + [self.target]
        self.list_edges_gauge()
        self.plaquette_operators()



    #ADDED: create_using=nx.DiGraph and periodic inverted
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

        graph_g = next(func)(self.n_sites[0],create_using=nx.DiGraph)
        for current_dim in self.n_sites[1:]:
            Gnew = next(func)(current_dim,create_using=nx.DiGraph)
            graph_g = cartesian_product( graph_g,Gnew)
        _graph = relabel_nodes(graph_g, flatten)
        self._graph = _graph

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
                for n_tmp in self.u_op_free #TODO: include gauss law option
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



        fig = plt.figure(figsize=(8, 6))


        color_map = []

        for node in self._graph:
            if self.dims==1:
                if node%2:
                    color_map.append("orange")
                else:
                    color_map.append("skyblue")
            else:
                if sum(node) % 2:
                    color_map.append("orange")
                else:
                    color_map.append("skyblue")

        connection = "arc3, rad = 0.16" if self.pbc else "arc3, rad = 0.0"

        if self.dims==1 or self.dims>3:
            for i, color_c in zip(self.graph_edges_system, edge_color_list):
                self._graph.add_edge(i[0], i[1], color=color_c)
            colors = nx.get_edge_attributes(self._graph, "color").values()




            nx.draw(
                self._graph,
                pos=nx.circular_layout(self._graph),#TODO find best layout for 1D or D>3
                node_color=color_map,
                with_labels=True,
                width=2,
                edge_color=colors,
                connectionstyle=connection,
                node_size=600*self.dims,
            )

        else:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(*np.array(self._graph.nodes).T, s=200, c=color_map)#,alpha=1)

            #Nodes labels
            for nds in np.array(self._graph.nodes):
                if self.dims==2:
                    ax.text(*nds-0.02,0,'('+",".join(map(str,nds))+')',fontsize=8)
                else:
                    ax.text(*nds-0.02,'('+",".join(map(str,nds))+')',fontsize=8)

            # Plot the edges
            arrow_options = dict(arrowstyle="-|>", mutation_scale=15,lw=1.5, connectionstyle=connection)

            for vizedge,col in zip(np.array(self._graph.edges),edge_color_list):
                if self.dims==2:
                    a = mplArrow3D(*vizedge.T,[0,0],ec=col,color=col,**arrow_options)
                else:
                    a = mplArrow3D(*vizedge.T,ec=col,color=col,**arrow_options)

                ax.add_artist(a)

            def _format_axes(ax):
                """Visualization options for the 3D axes."""
                for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                    dim.set_ticks([])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")

            _format_axes(ax)

        if self.pbc:
            bc_title = "PBC"
        else:
            bc_title = "OBC"

        fig.patch.set_facecolor('xkcd:white')
        fig.suptitle(
            'x'.join(map(str,self.n_sites))+f" Lattice:" + bc_title,
            fontsize=12,
            fontweight="bold",
            horizontalalignment="center",
        )

        if isinstance(savefig_dir, str):  # directory where to save figure
            fig.savefig(
                f"{savefig_dir}/system_"+"x".join(map(str,self.n_sites))+"_"
                + bc_title
                + f"_gausslaw{gauss_law_fig}.pdf",
                bbox_inches="tight",
                dpi=600,
            )

        plt.show()

    # # longest path in the graph #TODO: define most convenient path
    # def longest_simple_paths(self) -> List[List]:
    #     """Returns one of the longest path between two points (source and target)
    #     in the graph. Advantages when Jordan-Wigner chain is defined, since we
    #     can avoid long expressions with a lot of global terms and long Pauli-Z strings.

    #     Returns
    #     -------
    #     longest_paths:list of tuples.
    #         List of coordinates like [(0, 0), (0, 1),..] that corresponds to one of
    #         the longest paths.

    #     """
    #     longest_paths = []
    #     longest_path_len = 0
    #     for path in nx.all_simple_paths(
    #         self._graph, source=self.source, target=self.target
    #     ):
    #         if len(path) > longest_path_len:
    #             longest_path_len = len(path)
    #             longest_paths.clear()
    #             longest_paths.append(path)
    #         elif len(path) == longest_path_len:
    #             longest_paths.append(path)
    #     return longest_paths

    # # chain of elements for Jordan-Wigner function
    # def jw_chain_func(self):
    #     """Returns one of the longest path between two points (source and target)
    #     in the graph. Advantages when Jordan-Wigner chain is defined, since we
    #     can avoid long expressions with a lot of global terms and long Z strings.


    #     Returns
    #     -------
    #     longest_paths:list of tuples.
    #         List of coordinates like [(0, 0), (0, 1),..] that corresponds to one of
    #         the longest paths.

    #     """

    #     jw_chain = self.longest_simple_paths()[0]
    #     jw_chain = [(jw_chain[i], jw_chain[(i + 1)]) for i in range(len(jw_chain) - 1)]
    #     not_jwchain = [
    #         x
    #         for x in self._graph.edges()
    #         if x not in jw_chain and x not in [(t[1], t[0]) for t in jw_chain]
    #     ]
    #     not_jwchain = list(set([tuple(sorted(t)) for t in not_jwchain]))

    #     if (self.n_sitestot) - (len([k[0] for k in jw_chain]) + 1) != 0:
    #         warnings.warn(
    #         "Warning: Jordan-Wigner chain has missing sites. not long enough to reach every site."
    #         )

    #     self.jw_chain = jw_chain
    #     self.not_jwchain = not_jwchain

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

        #List of directions for the edges if D<=3 or more
        if self.dims<4:
            ax_direct = ["x","y","z"]
        else:
            ax_direct = list(map(chr, range(97, 123)))

        for u_elem, v_elem in self.graph_edges_system: #TODO check d in dagger U_d later
            op_chr_U = "U_"
            op_chr_E = "E_"
            coeff = 'none'

            if self.dims==1:
                edge_op1,edge_op2 = str(u_elem), str(v_elem)
                coeff = ax_direct[0]#Find in which direction the edge is

            else:
                edge_op1,edge_op2 = ''.join(map(str,u_elem)), ''.join(map(str,v_elem))
                #Find in which direction the edge is
                for index, (first, second) in enumerate(zip(u_elem,v_elem)):
                    if first != second:
                        coeff = ax_direct[index]

            #list of edges starting point and ending point
            list_edges.append((op_chr_E+edge_op1,op_chr_E+edge_op2))
            list_edges2_e_op.append((op_chr_E+edge_op1+coeff,op_chr_E+edge_op2+coeff))
            list_edges2_u_op.append((op_chr_U+edge_op1+coeff,op_chr_U+edge_op2+coeff))

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
                List of strings ("U_nx(y)"), i.e. plaquettes in terms of links operator

        """

        #TODO: test for dims>4 how to visualize it?
        plaq_list = []

        if len(self.n_sites)==1 and self.n_sites[0]==4:#only 1 dimension #TODO plaquette only if 4 sites pbc?
            if self.pbc is True:
                plaq_list = [list(self._graph.edges)]

        elif len(self.n_sites)>1:#more than 1 dimension
            for (a,b) in list(combinations(range(self.dims),2)):#change only 2 coordinates
                list_tmp = []
                for node in self._graph.nodes:

                    if isinstance(self.pbc,list): #check which directions are self. if self. iterable
                        op_tmp = lambda n,j: (n+1)%self.n_sites[j] if self.pbc[j] else n+1
                    elif isinstance(self.pbc,bool):#all directions pbc if pbc bool
                        op_tmp = lambda n,j: (n+1)%self.n_sites[j]
                    else:#no pbc
                        op_tmp = lambda n,j: n+1

                    tpl1=node

                    tpl2=tuple([op_tmp(n,j) if j==a else n for j,n in enumerate(node)])
                    if tpl2 not in self._graph.nodes:
                        continue

                    tpl3=tuple([op_tmp(n,i) if i==a or i==b else n for i,n in enumerate(node)])
                    if tpl3 not in self._graph.nodes:
                        continue

                    tpl4=tuple([op_tmp(n,j) if j==b else n for j,n in enumerate(node)])
                    if tpl4 not in self._graph.nodes:
                        continue

                    list_tmp.append([tpl1,tpl2,tpl3,tpl4])

                plaq_list.append(list_tmp)

            plaq_list = [item for sublist in plaq_list for item in sublist]

            plaq_list = [
                                [(p_tmp[i % len(p_tmp)], p_tmp[(i + 1) % len(p_tmp)])
                                for i in range(len(p_tmp))]
                                for p_tmp in plaq_list
                            ]


        if self.dims<4:
            ax_direct = ["x","y","z"]
        else:
            ax_direct = list(map(chr, range(97, 123)))

        list_plaq_u_op = []
        for i, p_tmp in enumerate(plaq_list):
            list_p = []
            j = 0
            for v_elem, u_elem in p_tmp:
                op_chr_U = "U_"
                coeff = 'none'

                if self.dims==1:
                    coeff = ax_direct[0]#Find in which direction the edge is

                else:
                    #Find in which direction the edge is
                    for index, (first, second) in enumerate(zip(u_elem,v_elem)):
                        if first != second:
                            coeff = ax_direct[index]

                vec = u_elem if j > 1 else v_elem  # U^dag after 2 edges: UUU^dagU^dag

                if self.dims==1:#TODO: not U^dag for 1D
                    list_p.append("U_" + str(v_elem)+ coeff)
                else:
                    list_p.append("U_" + ''.join(map(str,vec))+ coeff)
                j += 1

            list_plaq_u_op.append(list_p)

        self.plaq_list = plaq_list
        self.list_plaq_u_op = list_plaq_u_op


class mplArrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)