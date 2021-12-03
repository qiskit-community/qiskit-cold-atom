# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Fermionic basis tests."""

from qiskit.test import QiskitTestCase
from qiskit_cold_atom.fermions.fermion_circuit_solver import (
    FermionicState,
    FermionicBasis,
)


class TestFermionicBasis(QiskitTestCase):
    """Class to test the fermionic basis class."""

    def test_initialize(self):
        """Test the initialization of fermionic basis."""
        with self.subTest("particle and spin conserved"):
            basis = FermionicBasis(sites=3, n_particles=[1, 2, 3])
            self.assertEqual(basis.num_species, 3)
            self.assertEqual(basis.dimension, 9)
        with self.subTest("particle number not conserved"):
            basis = FermionicBasis(sites=3, n_particles=[1, 2, 3], particle_conservation=False)
            self.assertEqual(basis.dimension, 512)
        with self.subTest("particle number not conserved"):
            basis = FermionicBasis(sites=3, n_particles=[1, 2, 3], spin_conservation=False)
            self.assertEqual(basis.dimension, 84)

    def test_string(self):
        """Test the string representation."""
        basis = FermionicBasis(sites=2, n_particles=[1, 1])
        expect = "\n 0.   |0, 1>|0, 1>\n 1.   |0, 1>|1, 0>\n 2.   |1, 0>|0, 1>\n 3.   |1, 0>|1, 0>"
        self.assertEqual(str(basis), expect)

    def test_init_from_state(self):
        """Test the initialization from a fermionic state"""

        state = FermionicState([[0, 1, 1], [1, 0, 0]])
        with self.subTest("particle and spin conserved"):
            basis = FermionicBasis.from_state(state, spin_conservation=True)
            self.assertEqual(basis.num_species, 2)
            self.assertEqual(basis.dimension, 9)
        with self.subTest("particle number not conserved"):
            basis = FermionicBasis.from_state(state, spin_conservation=False)
            self.assertEqual(basis.dimension, 20)
        with self.subTest("particle number not conserved"):
            basis = FermionicBasis.from_state(
                state, spin_conservation=False, particle_conservation=False
            )
            self.assertEqual(basis.dimension, 64)

    def test_get_occupations(self):
        """Test the representation as flat occupations lists."""

        basis = FermionicBasis(sites=2, n_particles=[1, 1])
        self.assertEqual(
            basis.get_occupations(),
            [[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0]],
        )

    def test_index_of_measurement(self):
        """Test the allocation of the basis state index for a given measurement."""
        basis = FermionicBasis(sites=2, n_particles=[1, 1])
        self.assertEqual(basis.get_index_of_measurement("0110"), 1)
