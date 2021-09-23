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

""" Fermionic state tests."""

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase

from qiskit_cold_atom.exceptions import QiskitColdAtomError
from qiskit_cold_atom.fermions.fermion_circuit_solver import FermionicState


class TestFermionState(QiskitTestCase):
    """Class to test the fermion state class."""

    def test_initialize(self):
        """Test the initialization of fermionic states."""

        state = FermionicState([0, 1, 1, 0])

        self.assertEqual(state.sites, 4)
        self.assertEqual(state.num_species, 1)

        with self.assertRaises(QiskitColdAtomError):
            FermionicState([0, 2])

        with self.assertRaises(QiskitColdAtomError):
            FermionicState([[0, 1], [1, 0, 1]])

        state = FermionicState([[0, 1, 0], [1, 0, 1]])

        self.assertEqual(state.occupations_flat, [0, 1, 0, 1, 0, 1])
        self.assertEqual(state.num_species, 2)

    def test_string(self):
        """Test the string representation."""

        state = FermionicState([0, 1])

        self.assertEqual(str(state), "|0, 1>")

        state = FermionicState([[0, 1], [1, 0]])
        self.assertEqual(str(state), "|0, 1>|1, 0>")

    def test_occupations(self):
        """Test that to get the fermionic occupations."""
        state = FermionicState([0, 1])

        self.assertEqual(state.occupations, [[0, 1]])

    def test_from_flat_list(self):
        """Test the creation of fermionic states from flat lists."""

        state = FermionicState.from_total_occupations([0, 1, 1, 0], 2)
        self.assertEqual(state.occupations, [[0, 1], [1, 0]])

        with self.assertRaises(QiskitColdAtomError):
            FermionicState.from_total_occupations([0, 1, 1, 0], 3)

        state = FermionicState.from_total_occupations([0, 1, 1, 0], 1)
        self.assertEqual(state.occupations, [[0, 1, 1, 0]])

    def test_from_initial_state(self):
        """Test that we can load an initial state from a circuit."""

        circ = QuantumCircuit(4)
        circ.load_fermions(0)
        circ.load_fermions(2)

        state = FermionicState.initial_state(circ, 2)
        self.assertEqual(state.occupations, [[1, 0], [1, 0]])
        self.assertEqual(state.occupations_flat, [1, 0, 1, 0])

        state = FermionicState.initial_state(circ, 1)
        self.assertEqual(state.occupations, [[1, 0, 1, 0]])
        self.assertEqual(state.occupations_flat, [1, 0, 1, 0])
