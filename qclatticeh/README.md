**WARNING** This is an experimental branch.

## Package Version

This branch contains a packaged version of QC_lattice_H for future distribution.

### Building the package

To test this package, first install `hatch` in your python environment:

```console
pip install hatch
```

Then, from the QC_lattice_H root directory, build the package wheels and sdist locally:

```console
hatch build
```

Finally, test the installation (again from the root directory, where `pyproject.toml` can be found)

```console
pip install .
```

To also include optional dependencies (`scipy`, `qiskit`), use the command below (**Warning:** if you are using qiskit 1.x, do **NOT!** run the command below, and use your own manually installed version of qiskit:

```console
pip install .[all]
```


### Using the package

To use the package, you can now import the package:

```python
import qclatticeh
```

#### Replacing previous code

To achieve the same functionality, you can replace the lines below:

```python
import sys
sys.path.append("../") # go to parent dir
from Hamiltonian_QED_sym import HamiltonianQED_sym
from Hamiltonian_QED_oprt import HamiltonianQED_oprt
from HC_Lattice import HCLattice
from Ansaetze import Ansatz
```

with the package version:

```python
from qclatticeh.lattice import HCLattice
from qclatticeh.hamiltonian import HamiltonianQED_sym, HamiltonianQED_oprt
from qclatticeh.circuits import Ansatz
```


### Optional dependencies

