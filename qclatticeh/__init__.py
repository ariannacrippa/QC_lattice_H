"""
qclatticeh: Lattice QED Hamiltonian for Quantum Computing

"""

__version__ = "0.1.0-b"


import importlib

def load_module(module_name):
    try:
        return importlib.import_module(f'.{module_name}', package=__name__)
    except ImportError as e:
        raise ImportError(f"Failed to import {module_name} in {__name__}")

def __getattr__(name):
    if name in ['lattice', 'hamiltonian', 'circuits']:
        return load_module(name)
    else:
        raise AttributeError(f"module {__name__} has no implemented submodule {name}")

__all__ = ['lattice', 'hamiltonian', 'circuits']
