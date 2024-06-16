import pytest
from HC_Lattice import HCLattice
from iteration_utilities import duplicates


# Tests based on the Notebook "class_HC_lattice_test.ipynb"
test_lattices = [
        ([3, 2, 2], [True, False, False]),
        ([3, 2], [True, False]),
        #([3], [True]),
    ]
@pytest.mark.parametrize("n_sites, periodic", test_lattices)
def test_class_initialization(n_sites, periodic):
    lattice = HCLattice(n_sites, periodic)
    assert lattice.n_sites == n_sites
    assert lattice.pbc == periodic
    assert lattice.dims == len(lattice.n_sites)
    assert lattice.dims == len(lattice.pbc)




test_dimensions = [
        [3, 2, 1],
        [3, 1, 2],
        [1, 3, 2],
        [3, 2, 1, 1],
        [3, 1, 2, 1],
        [1, 3, 2, 1],
        [3, 1, 1, 2],
        [1, 3, 1, 2],
        [1, 1, 3, 2]
    ]
@pytest.mark.parametrize("n_sites", test_dimensions)
def test_useless_dimension_reduction(n_sites):
    ref_lattice = HCLattice([3, 2])
    lattice = HCLattice(n_sites)
    assert lattice.n_sites == ref_lattice.n_sites
    

@pytest.fixture
def lattice_322():
    return HCLattice([3, 2, 2], [True, False, False])


def test_class_attributes_dim(lattice_322):
    assert lattice_322.n_sites == [3, 2, 2]
    assert lattice_322.pbc == [True, False, False]
    assert lattice_322.dims == 3


def test_class_attributes_list_edges2_e_op(lattice_322):
    assert lattice_322.list_edges2_e_op == ['E_000x', 'E_000y', 'E_000z',
                                            'E_001x', 'E_001y', 'E_010x',
                                            'E_010z', 'E_011x', 'E_100x',
                                            'E_100y', 'E_100z', 'E_101x',
                                            'E_101y', 'E_110x', 'E_110z',
                                            'E_111x', 'E_200x', 'E_200y',
                                            'E_200z', 'E_201x', 'E_201y',
                                            'E_210x', 'E_210z', 'E_211x']


def test_class_attributes_list_plaq_u_op(lattice_322):
    assert lattice_322.list_plaq_u_op == [['U_000x', 'U_100y', 'U_010x', 'U_000y'],
                                         ['U_001x', 'U_101y', 'U_011x', 'U_001y'],
                                         ['U_100x', 'U_200y', 'U_110x', 'U_100y'],
                                         ['U_101x', 'U_201y', 'U_111x', 'U_101y'],
                                         ['U_200x', 'U_000y', 'U_210x', 'U_200y'],
                                         ['U_201x', 'U_001y', 'U_211x', 'U_201y'],
                                         ['U_000x', 'U_100z', 'U_001x', 'U_000z'],
                                         ['U_010x', 'U_110z', 'U_011x', 'U_010z'],
                                         ['U_100x', 'U_200z', 'U_101x', 'U_100z'],
                                         ['U_110x', 'U_210z', 'U_111x', 'U_110z'],
                                         ['U_200x', 'U_000z', 'U_201x', 'U_200z'],
                                         ['U_210x', 'U_010z', 'U_211x', 'U_210z'],
                                         ['U_000y', 'U_010z', 'U_001y', 'U_000z'],
                                         ['U_100y', 'U_110z', 'U_101y', 'U_100z'],
                                         ['U_200y', 'U_210z', 'U_201y', 'U_200z']]


def test_class_attributes_jw_chain(lattice_322):
    assert lattice_322.jw_chain == [((0, 0, 0), (1, 0, 0)),
                                     ((1, 0, 0), (2, 0, 0)),
                                     ((2, 0, 0), (2, 1, 0)),
                                     ((2, 1, 0), (1, 1, 0)),
                                     ((1, 1, 0), (0, 1, 0)),
                                     ((0, 1, 0), (0, 1, 1)),
                                     ((0, 1, 1), (1, 1, 1)),
                                     ((1, 1, 1), (2, 1, 1)),
                                     ((2, 1, 1), (2, 0, 1)),
                                     ((2, 0, 1), (1, 0, 1)),
                                     ((1, 0, 1), (0, 0, 1))]


def test_class_attributes_jw_sites(lattice_322):
    assert lattice_322.jw_sites == [(0, 0, 0),
                                     (1, 0, 0),
                                     (2, 0, 0),
                                     (2, 1, 0),
                                     (1, 1, 0),
                                     (0, 1, 0),
                                     (0, 1, 1),
                                     (1, 1, 1),
                                     (2, 1, 1),
                                     (2, 0, 1),
                                     (1, 0, 1),
                                     (0, 0, 1)]


def test_class_attributes_duplicate_plaquettes(lattice_322):
    assert len(list(duplicates(lattice_322.plaq_list, key=tuple))) == 0


