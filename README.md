# Lattice QED Hamiltonian for Quantum Computing

In this repository one can write the QED Hamiltonian for a 1,2 or 3D lattice with staggered fermions, i.e. Kogut and Susskind formulation. The code is compatible with exact diagonalisation libraries and Qiskit library.

## Python scripts
**'HC_Lattice.py'**
A python code that builds a lattice in generic N dimensions, with periodic or open boundary conditions.
It finds the set of links and sites, build plaquettes and chain for Jordan_Wigner definition (version up to 3D).

**'Hamiltonian_QED_sym.py'**
A python code that builds a symbolic expression of QED Hamiltonian N-dimensional lattice, both with open and periodic boundary conditions. The formulation considered is from Kogut and Susskind and the Gauss’ law is applied.
By doing so, we will get a gauge invariant system, reduce the number of dynamical links needed
for the computation, and have a more resource-efficient Hamiltonian suitable for a wide range of
quantum hardware.

**'Hamiltonian_QED_oprt.py'**
A python code that imports the Hamiltonian from symbolic expression and build the operator form (sparse matrices or PauliOp, suitable for qiskit quantum circuits).
It considers two types of encoding: 'ed' returns sparse matrix, 'gray' with option sparse=False it returns PauliOp expression, otherwise a sparse matrix.

(For tests see: [class_H_QED_test_sym_oprt.ipynb](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/class_H_QED_test_sym_oprt.ipynb))

**'Ansaetze.py'**
Ansaetze proposal of variational circuit for Gray encoding (for gauge fields) and zero-charge sector (for femrionic d.o.f.).


## Example

Let us consider a 2x2 OBC system as in following figure:

![alt text](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/system_2x2_OBC_gausslawTrue.png)

where the black arrow represent the gauge field that remains dynamical after Gauss law is applied, i.e.

$$\eqalign{- E_{00x} - E_{00y} - q_{00} &= 0 \\
E_{00y} - E_{01x} - q_{01} &= 0 \\
E_{00x} - E_{10y} - q_{10} &= 0 \\
E_{01x} + E_{10y} - q_{11} &= 0 \\
q_{00} + q_{01} + q_{10} + q_{11} &= 0.
}$$

After this step, the Hamiltonian will be:

$$
H_{E} = \frac{g^{2} \left(E_{10y}^{2} + \left(- E_{10y} + q_{11}\right)^{2} + \left(E_{10y} + q_{10}\right)^{2} + \left(- E_{10y} + q_{01} + q_{11}\right)^{2}\right)}{2}
$$

for the electric part,

$$
H_{B} = - \frac{U_{10y} + h.c.}{2 g^{2}}
$$

for magnetic. If matter fields are considered, then we have a mass term

$$
H_{m} = m \left(\Phi_{1}^{\dagger} \Phi_{1} - \Phi_{2}^{\dagger} \Phi_{2} + \Phi_{3}^{\dagger} \Phi_{3} - \Phi_{4}^{\dagger} \Phi_{4}\right)
$$

and a kinetic term

$$
H_{K} = \Omega \left(0.5 i \left(- h.c.(x) + \Phi_{1}^{\dagger} \Phi_{2} + \Phi_{4}^{\dagger} \Phi_{3}\right) - 0.5 \left(h.c.(y) + \Phi_{1}^{\dagger} \Phi_{4} - \Phi_{2}^{\dagger} U_{10y}^{\dagger} \Phi_{3}\right)\right).
$$

One can visualize the Hamiltonian and then decide which encoding to use and if the final expression must be written in terms of Qiskit's Pauli operators.

It is also possible to put static charges on the sites and study the static potential.

## Gray circuit

After discretizing and truncating the $U(1)$ gauge group to the discrete group $\mathbb{Z}_{2l+1}$, the gauge fields can be represented in the electric basis as
\begin{subequations}
\begin{align}
    \hat{E} &= \sum_{i=-l}^l i \ket{i}_{\text{ph}}\bra{i}_{\text{ph}}, \\
    \hat{U} &= \sum_{i=-l}^{l-1} \ket{i+1}_{\text{ph}}\bra{i}_{\text{ph}},\\
    \hat{U^\dagger} &= \sum_{i=-l+1}^l \ket{i-1}_{\text{ph}}\bra{i}_{\text{ph}}.
\end{align}
\end{subequations}
For numerical calculations, it is advantageous to employ a suitable encoding that accurately represents the physical values of the gauge fields.
In this work, we consider the \textit{Gray encoding} .
For the truncation $l=1$, we can use the circuit in the following Figure to represent a gauge field.

![alt text](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/gray_circuit_l1.png)

The action of the circuit is straightforward: starting from the state $\ket{00}$, setting both parameters $\theta_1$ and $\theta_2$ to zero allows for the exploration of the physical state $\ket{-1}_{\text{ph}}$. The introduction of a non-zero value for $\theta_1$ allows the state to change to $\ket{01}$, which represents the  \textit{vacuum state} $\ket{0}_{\text{ph}}$, with a certain probability. A complete rotation occurs if $\theta_1=\pi$, resulting in the exclusive presence of the second state with a probability of 1.0. Subsequently, the second controlled gate operates only when the first qubit is $\ket{1}$, limiting the exploration to $\ket{11}$ (i.e., $\ket{1}_{\text{ph}}$) and excluding $\ket{10}$.

Circuits for larger truncations ($l=3,7,15$) are:

![alt text](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/gray_circuit_l3.png)
![alt text](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/gray_circuit_l7.png)
![alt text](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/gray_circuit_l15.png)


Ref: ![alt text](https://arxiv.org/abs/2404.17545)

