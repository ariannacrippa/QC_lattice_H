[![DOI](https://zenodo.org/badge/642751290.svg)](https://zenodo.org/badge/latestdoi/642751290)

# Lattice QED Hamiltonian for Quantum Computing

This repository contains useful Python functions and classes for seamlessly designing and obtaining Quantum Electrodynamic (QED) Hamiltonians for 1D, 2D or 3D lattices with staggered fermionic degrees of freedom, i.e.,  Kogut and Susskind formulation[^1]. 

The implementation is useful for carrying out research on QED quantum simulation assisted by quantum computing and quantum algorithms, as well as for analysis and comparison with known exact-diagonalization (ED) methods. In turn, the classes in this module are compatible with exact diagonalization libraries and the Qiskit library.

**Related work:** 
- [Analysis of the confinement string in (2 + 1)-dimensional Quantum Electrodynamics with a trapped-ion quantum computer](https://arxiv.org/abs/2411.05628)
- [Towards determining the (2+1)-dimensional Quantum Electrodynamics running coupling with Monte Carlo and quantum computing methods](https://arxiv.org/abs/2404.17545)

## Installation

This source code is now available at [PyPI](https://pypi.org/projects/qclatticeh) as a package for quick installation and usage. One can simply `pip install` it with the command below:

```console
pip install qclatticeh
```

Note: if you are not using a python environment (recommended), you may need to use the `pip3` command instead of `pip` in MacOS and Linux.


### Manual Installation through Github

It may be useful to some researchers to adapt the current package to their own models. In that case, a manual installation may be easier for making changes in the hamiltonian formulation.

#### Clone
The first step is to download the latest stable release of this package below and unzip de file.

[Latest Release](https://github.com/ariannacrippa/QC_lattice_H/releases)

Altenatively, you can clone this repo for development builds:

```console
git clone https://github.com/ariannacrippa/QC_lattice_H.git
```

#### Virtual Environment
Before installing this package manually, make sure you have made a dedicated Python virtual environment and installed all required dependencies.

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

#### Dependencies


- General Dependencies:
    `numpy`, `matplotlib`, `networkx`, `sympy`, `iteration_utilities`

- For Exact Diagonalization:
    `scipy`

- For Quantum Computation:
    `qiskit`, `qiskit.quantum_info`
    
It is possible to install all dependencies with the command below:

```bash
pip install numpy matplotlib networkx sympy scipy qiskit iteration_utilities
```

#### Manual installation
You can use pip to carry out the manual installation. With your python environment activated, run:

```console
pip install .
```


## Usage

This module consists of three main Python classes found in three separate submodules, as explained below, each focused on a particular component of QED Hamiltonian creation.

### Important Submodules
- **'lattice'** [link to source code](https://github.com/ariannacrippa/QC_lattice_H/blob/main/qclatticeh/lattice)

-Contains a python class '`HCLattice`' that builds a lattice in generic N dimensions, with periodic or open boundary conditions.
It finds the set of links and sites, builds plaquettes and chains for Jordan-Wigner[^2] definition (version up to 3D). 


- **'hamiltonian'** [link to source code](https://github.com/ariannacrippa/QC_lattice_H/blob/main/qclatticeh/hamiltonian)

-Has a python class '`HamiltonianQED_sym`' that builds a symbolic expression of QED Hamiltonian N-dimensional lattice, both with open and periodic boundary conditions. The formulation considered is from Kogut and Susskind and the Gauss’ law can be applied.
By doing so, it results in a gauge-invariant system, reducing the number of dynamical links needed for the computation, leading to a resource-efficient Hamiltonian.
It is also possible to consider a formulation for a magnetic basis[^3].


-Also contains a '`HamiltonianQED_oprt`' class. It imports the Hamiltonian from symbolic expression and builds its respective operator form (sparse matrices or PauliOp, suitable for qiskit quantum circuits).
It considers two types of encoding: `ed` returns sparse matrix; `gray` with option sparse=False returns PauliOp expression. Otherwise, a sparse matrix is returned.


- **'circuits'** [link to source code](https://github.com/ariannacrippa/QC_lattice_H/blob/main/qclatticeh/circuits)

-Contains an '`Ansatz`' class. Consists of ansaetze proposals of variational circuits for Gray encoding (for gauge fields) and zero-charge sector (for fermionic degrees of freedom).

### Importing classes
As shorthand, you can import all the classes into your python project as below.


```python
from qclatticeh.lattice import HCLattice
from qclatticeh.hamiltonian import HamiltonianQED_sym, HamiltonianQED_oprt
from qclatticeh.circuits import Ansatz
```

### Migration Guide
In case you were using older versions of this source code before packaging, you may have imported the same python classes as seen below:



```python
import sys
sys.path.append("../") # go to parent dir
from Hamiltonian_QED_sym import HamiltonianQED_sym
from Hamiltonian_QED_oprt import HamiltonianQED_oprt
from HC_Lattice import HCLattice
from Ansaetze import Ansatz
```

You can simply replace the imports with the new package import commands, and most of the old functionality and codebase should work successfully:

```python
from qclatticeh.lattice import HCLattice
from qclatticeh.hamiltonian import HamiltonianQED_sym, HamiltonianQED_oprt
from qclatticeh.circuits import Ansatz
```



### Examples

For code examples illustrating a typical workflow with this module, please refer to the `notebooks` folder and the Jupyter Notebooks examples there:

- [Test class Hamiltonian symbolic and operators](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/class_H_QED_test_sym_oprt.ipynb)
- [Test class lattices](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/class_HC_lattice_test.ipynb)
- [Test class Ansaetze](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/class_ansaetze.ipynb)
- [Qiskit VQE example](https://github.com/ariannacrippa/QC_lattice_H/blob/main/notebooks/qclatticeh_vqe_example.ipynb)


For an example of Hamiltonian design, let us consider a 2x2 OBC system as in the following figure:
![Image](https://github.com/user-attachments/assets/29b41073-8d9b-4fbf-ad64-299eeeb491a8)

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

After discretizing and truncating the U(1) gauge group to the discrete group Z(2l+1), the gauge fields can be represented in the electric basis as

$$
\hat{E} = \sum_{i=-l}^l i \ket{i}\bra{i},
$$

$$
\hat{U} = \sum_{i=-l}^{l-1} \ket{i+1}\bra{i},
$$

$$
\hat{U^\dagger} = \sum_{i=-l+1}^l \ket{i-1}\bra{i},
$$

$$\text{where}\ket{i}=\ket{i}_{\text{ph}} $$.

For numerical calculations, it is advantageous to employ a suitable encoding that accurately represents the physical values of the gauge fields.
In this work, we consider the Gray encoding.
For the truncation `l=1`, we can use the circuit in the following Figure to represent a gauge field.

![Image](https://github.com/user-attachments/assets/140de3cc-a539-4a54-9035-36f7dde6f853)

The action of the circuit is straightforward: starting from the state |00>, setting both parameters θ1 and θ2 to zero allows for the exploration of the physical state |-1>. The introduction of a non-zero value for θ1 allows the state to change to |01>, which represents the physical vacuum state |0>, with a certain probability. A complete rotation occurs if θ1 = π, resulting in the exclusive presence of the second state with a probability of 1.0. Subsequently, the second controlled gate operates only when the first qubit is |1>, limiting the exploration to |11> (i.e., physical |1>) and excluding |10>.

Circuits for larger truncations (`l=3,7,15`) are:

![Image](https://github.com/user-attachments/assets/cd812b92-2c88-4516-965d-8a3fd50e4dba)

![Image](https://github.com/user-attachments/assets/287d3ba4-c601-48c2-9cc7-1b5bb9f0388d)

![Image](https://github.com/user-attachments/assets/f42b9280-74a5-430a-b90f-104a964e0028)


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


