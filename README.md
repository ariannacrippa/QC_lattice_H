[![DOI](https://zenodo.org/badge/642751290.svg)](https://zenodo.org/badge/latestdoi/642751290)

# Lattice QED Hamiltonian for Quantum Computing

This repository contains useful Python functions and classes for seamlessly designing and obtaining Quantum Electrodynamic (QED) Hamiltonians for 1D, 2D or 3D lattices with staggered fermionic degrees of freedom, i.e.,  Kogut and Susskind formulation[^1]. 

The implementation is useful for carrying out research on QED quantum simulation assisted by quantum computing and quantum algorithms, as well as for analysis and comparison with known exact-diagonalization (ED) methods. In turn, the classes in this module are compatible with exact diagonalization libraries and the Qiskit library.

**Related work:** 
- [Analysis of the confinement string in (2 + 1)-dimensional Quantum Electrodynamics with a trapped-ion quantum computer](https://arxiv.org/abs/2411.05628)
- [Towards determining the (2+1)-dimensional Quantum Electrodynamics running coupling with Monte Carlo and quantum computing methods](https://arxiv.org/abs/2404.17545)


## Installation
Before using this code, make sure you have made a dedicated Python virtual environment and installed all required dependencies.

### Virtual Environment
There are many distinct ways of creating Python environments which might be suitable for different applications, e.g., Anaconda, Miniconda, PyEnv, Venv, and others. Use the one that best suits your needs.

For convenience purposes, we provide an example of how to create and activate a simple environment using Venv:

Unix Systems (Zsh and Bash):
```bash
python3 -m venv qc_lattice_h && source qc_lattice_h/bin/activate
```

Windows:
```bash
python -m venv qc_lattice_h && qc_lattice_h/Scripts/activate
```

To leave this environment, simply run '`deactivate`'.

### Dependencies


- General Dependencies:
    `numpy`, `matplotlib`, `networkx`, `sympy`, `iteration_utilities`

- For Exact Diagonalization:
    `scipy`

- For Quantum Computation:
    `qiskit`, `qiskit.quantum_info` (Version 1.0)
    
It is possible to install all dependencies with the command below:

```bash
pip install numpy matplotlib networkx sympy scipy qiskit iteration_utilities
```

## Usage

This module consists of three main Python classes found in three separate files, as explained below, each focused on a particular component of QED Hamiltonian creation.

### File Structure
- **'HC_Lattice.py'** [link](https://github.com/ariannacrippa/QC_lattice_H/blob/main/qclatticeh/lattice/HC_Lattice.py)

-Contains a python class '`HCLattice`' that builds a lattice in generic N dimensions, with periodic or open boundary conditions.
It finds the set of links and sites, builds plaquettes and chains for Jordan-Wigner[^2] definition (version up to 3D). 


- **'Hamiltonian_QED_sym.py'** [link](https://github.com/ariannacrippa/QC_lattice_H/blob/main/qclatticeh/hamiltonian/Hamiltonian_QED_sym.py)

-Has a python class '`HamiltonianQED_sym`' that builds a symbolic expression of QED Hamiltonian N-dimensional lattice, both with open and periodic boundary conditions. The formulation considered is from Kogut and Susskind and the Gauss’ law can be applied.
By doing so, it results in a gauge-invariant system, reducing the number of dynamical links needed for the computation, leading to a resource-efficient Hamiltonian.
It is also possible to consider a formulation for a magnetic basis[^3].

- **'Hamiltonian_QED_oprt.py'** [link](https://github.com/ariannacrippa/QC_lattice_H/blob/main/qclatticeh/hamiltonian/Hamiltonian_QED_oprt.py)

-Contains a '`HamiltonianQED_oprt`' class. It imports the Hamiltonian from symbolic expression and builds its respective operator form (sparse matrices or PauliOp, suitable for qiskit quantum circuits).
It considers two types of encoding: `ed` returns sparse matrix; `gray` with option sparse=False returns PauliOp expression. Otherwise, a sparse matrix is returned.


- **'Ansaetze.py'** [link](https://github.com/ariannacrippa/QC_lattice_H/blob/main/qclatticeh/circuits/Ansaetze.py)

-Contains an '`Ansatz`' class. Consists of ansaetze proposals of variational circuits for Gray encoding (for gauge fields) and zero-charge sector (for fermionic degrees of freedom).

### Importing classes

To integrate this implementation into your project, one suggestion is to include the Python files in your project folder, and import the classes as shown below:

```python
from Hamiltonian_QED_sym import HamiltonianQED_sym
from Hamiltonian_QED_oprt import HamiltonianQED_oprt
from HC_Lattice import HCLattice
from Ansaetze import *
```

Alternatively, one could clone the full repository (or include it as a git submodule), in which case the files would be saved in a `QC_lattice_H` subfolder. Make sure to update your code to reflect the different path, either by using a `sys.path.append()` call or by adding a prefix to your imports: `HC_Lattice` -> `QC_lattice_H.HC_Lattice`. 


### Examples

For code examples illustrating a typical workflow with this module, please refer to the `notebooks` folder and the Jupyter Notebooks examples there:

- [Test class Hamiltonian symbolic and operators](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/class_H_QED_test_sym_oprt.ipynb)
- [Test class lattices](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/class_HC_lattice_test.ipynb)
- [Test class Ansaetze](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/class_ansaetze.ipynb)
- [Qiskit VQE example](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/qclatticeh_vqe_example.ipynb)


For an example of Hamiltonian design, let us consider a 2x2 OBC system as in the following figure:

<img src="https://github.com/ariannacrippa/QC_lattice_H/blob/main/Images/system_2x2_OBC_gausslawTrue.png" width="400" height="400">

where the black arrow represents the gauge field that remains dynamical after Gauss law is applied, i.e.

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

To encode the fermions for a quantum circuit implementation we consider the Jordan-Wigner formulation with:

$$
\begin{aligned}\Phi_j &= \prod_{k} (-i\sigma^z_k) \sigma^{+}_j , \quad \text{where } k < j,
\end{aligned}
$$

$$
\begin{aligned}\Phi_j^{\dagger}&=  \prod_{k} (i\sigma^z_k) \sigma^{-}_j, \quad \text{where } k < j.
\end{aligned}
$$


The dynamical charges can be written as

$$q_{\vec{n}} =  \Phi^{\dagger} \Phi-\frac{1}{2}(1+(-1)^{n_{x} +n_{y} + 1}) \ \to \ \ \begin{cases}			\frac{I-\sigma_{z}}{2}, & \text{if $\vec{n}$ even}\\
            -\frac{I+\sigma_{z}}{2}, & \text{if $\vec{n}$ odd}
		 \end{cases}$$

## Gray circuit

After discretizing and truncating the U(1) gauge group to the discrete group $`\mathbb{Z}_{2l+1}`$, the gauge fields can be represented in the electric basis as

$$\eqalign{
\hat{E} &= \sum_{i=-l}^l i \ket{i}\bra{i}, \\
\hat{U} &= \sum_{i=-l}^{l-1} \ket{i+1}\bra{i}, \\
\hat{U^\dagger} &= \sum_{i=-l+1}^l \ket{i-1}\bra{i},
}$$

where $`\ket{i}=\ket{i}_{\text{ph}}`$.

For numerical calculations, it is advantageous to employ a suitable encoding that accurately represents the physical values of the gauge fields.
In this work, we consider the Gray encoding.
For the truncation $`l=1`$, we can use the circuit in the following Figure to represent a gauge field.

<img src="https://github.com/ariannacrippa/QC_lattice_H/blob/main/Images/gray_circuit_l1.png" width="400" height="200">

The action of the circuit is straightforward: starting from the state $`\ket{00}`$, setting both parameters $`\theta_1`$ and $`\theta_2`$ to zero allows for the exploration of the physical state $`\ket{-1}_{\text{ph}}`$. The introduction of a non-zero value for $`\theta_1`$ allows the state to change to $`\ket{01}`$, which represents the vacuum state $`\ket{0}_{\text{ph}}`$, with a certain probability. A complete rotation occurs if $`\theta_1=\pi`$, resulting in the exclusive presence of the second state with a probability of 1.0. Subsequently, the second controlled gate operates only when the first qubit is $`\ket{1}`$, limiting the exploration to $`\ket{11}`$ (i.e., $`\ket{1}_{\text{ph}}`$) and excluding $`\ket{10}`$.

Circuits for larger truncations ($`l=3,7,15`$) are:

<img src="https://github.com/ariannacrippa/QC_lattice_H/blob/main/Images/gray_circuit_l3.png" width="400" height="200">
<img src="https://github.com/ariannacrippa/QC_lattice_H/blob/main/Images/gray_circuit_l7.png" width="400" height="200">
<img src="https://github.com/ariannacrippa/QC_lattice_H/blob/main/Images/gray_circuit_l15.png" width="400" height="200">


## Fermionic circuit
For simulating the fermionic degrees of freedom we employ the `iSWAP` gate, see Ref.[^4].


## Feedback and Bugs
If any bugs related to installation, usage and workflow are encountered, and in case of suggestions for improvements and new features, please open a [New Issue](https://github.com/ariannacrippa/QC_Lattice_H/issues/new)

## Contributing
To assist with potential contributions, please contact the lead developer Arianna Crippa before submitting a Pull Request.

### License
[MIT licence](https://github.com/ariannacrippa/QC_lattice_H/blob/main/LICENSE)
## References
[^1]: [Lattice fermions L Susskind - Physical Review D, 1977 - APS](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.16.3031)
[^2]: [P. Jordan and E. P. Wigner, Über das paulische äquivalenzverbot, in The Collected Works of Eugene Paul Wigner (Springer, New York, 1993), pp. 109–129](https://link.springer.com/chapter/10.1007/978-3-662-02781-3_9)
[^3]: [JF Haase, L Dellantonio, A Celi, D Paulson, A Kan, K Jansen, CA Muschik Quantum, 2021•quantum-journal.org](https://quantum-journal.org/papers/q-2021-02-04-393/)
[^4]: [D. Paulson, L. Dellantonio, J. F. Haase, A. Celi, A. Kan, A. Jena, C. Kokail, R. van Bijnen, K. Jansen, P. Zoller, C. A. Muschik, arXiv:2008.09252](https://arxiv.org/abs/2008.09252)


