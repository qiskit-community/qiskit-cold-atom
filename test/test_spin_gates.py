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

""" Spin gate tests."""

import numpy as np
from scipy.linalg import expm

from qiskit.test import QiskitTestCase
from qiskit import QuantumCircuit
from qiskit_nature.operators.second_quantization import SpinOp
from qiskit_cold_atom.spins.spin_circuit_solver import SpinCircuitSolver
from qiskit_cold_atom.spins import SpinSimulator
from qiskit_cold_atom.spins.spins_gate_library import (
    RLXGate,
    RLYGate,
    RLZGate,
    RLZ2Gate,
    RLxLyGate,
)


class TestSpinGates(QiskitTestCase):
    """Tests for the spin hardware gates"""

    def setUp(self):
        super().setUp()
        self.backend = SpinSimulator()
        self.spin = 3 / 2
        self.solver = SpinCircuitSolver(spin=self.spin)

    def test_lx_gate(self):
        """check matrix form of the lx gate"""
        omega = np.pi / 2
        circ = QuantumCircuit(1)
        circ.append(RLXGate(omega), qargs=[0])
        # add gate to circuit via the @add_gate-decorated method
        circ_decorated = QuantumCircuit(1)
        circ_decorated.rlx(omega, 0)

        for circuit in [circ, circ_decorated]:
            unitary = self.backend.run(circuit, spin=self.spin).result().get_unitary()
            self.assertTrue(
                np.allclose(
                    unitary,
                    expm(
                        -1j
                        * omega
                        * np.array(
                            [
                                [0.0, np.sqrt(3) / 2, 0, 0],
                                [np.sqrt(3) / 2, 0, 1, 0],
                                [0, 1, 0, np.sqrt(3) / 2],
                                [0, 0, np.sqrt(3) / 2, 0],
                            ]
                        )
                    ),
                )
            )

    def test_lxly_gate(self):
        """check matrix form of the lxly gate"""
        omega = np.pi
        circ = QuantumCircuit(2)
        circ.append(RLxLyGate(omega), qargs=[0, 1])

        # add gate to circuit via the @add_gate-decorated method
        circ_decorated = QuantumCircuit(2)
        circ_decorated.rlxly(omega, [0, 1])

        for circuit in [circ, circ_decorated]:
            unitary = self.backend.run(circuit, spin=1 / 2).result().get_unitary()
            self.assertTrue(
                np.allclose(
                    unitary,
                    [
                        [1, 0, 0, 0],
                        [0, 0, -1j, 0],
                        [0, -1j, 0, 0],
                        [0, 0, 0, 1],
                    ],
                )
            )

    def test_ly_gate(self):
        """check matrix form of the ly gate"""
        omega = np.pi / 2
        circ = QuantumCircuit(1)
        circ.append(RLYGate(omega), qargs=[0])
        # add gate to circuit via the @add_gate-decorated method
        circ_decorated = QuantumCircuit(1)
        circ_decorated.rly(omega, 0)

        for circuit in [circ, circ_decorated]:
            unitary = self.backend.run(circuit, spin=self.spin).result().get_unitary()
            self.assertTrue(
                np.allclose(
                    unitary,
                    expm(
                        -1j
                        * omega
                        * np.array(
                            [
                                [0.0, -1j * np.sqrt(3) / 2, 0, 0],
                                [1j * np.sqrt(3) / 2, 0, -1j, 0],
                                [0, 1j, 0, -1j * np.sqrt(3) / 2],
                                [0, 0, 1j * np.sqrt(3) / 2, 0],
                            ]
                        )
                    ),
                )
            )

    def test_lz_gate(self):
        """check matrix form of the lz gate"""
        delta = np.pi / 2
        circ = QuantumCircuit(1)
        circ.append(RLZGate(delta), qargs=[0])
        # add gate to circuit via the @add_gate-decorated method
        circ_decorated = QuantumCircuit(1)
        circ_decorated.rlz(delta, 0)

        for circuit in [circ, circ_decorated]:
            unitary = self.backend.run(circuit, spin=self.spin).result().get_unitary()
            self.assertTrue(
                np.allclose(
                    unitary,
                    expm(
                        -1j
                        * delta
                        * np.array(
                            [
                                [3 / 2, 0, 0, 0],
                                [0, 1 / 2, 0, 0],
                                [0, 0, -1 / 2, 0],
                                [0, 0, 0, -3 / 2],
                            ]
                        )
                    ),
                )
            )

    def test_lz2_gate(self):
        """check matrix form of the lz2 gate"""
        chi = np.pi / 2
        circ = QuantumCircuit(1)
        circ.append(RLZ2Gate(chi), qargs=[0])
        # add gate to circuit via the @add_gate-decorated method
        circ_decorated = QuantumCircuit(1)
        circ_decorated.rlz2(chi, 0)

        for circuit in [circ, circ_decorated]:
            unitary = self.backend.run(circuit, spin=self.spin).result().get_unitary()
            self.assertTrue(
                np.allclose(
                    unitary,
                    expm(
                        -1j
                        * chi
                        * np.array(
                            [
                                [3 / 2, 0, 0, 0],
                                [0, 1 / 2, 0, 0],
                                [0, 0, -1 / 2, 0],
                                [0, 0, 0, -3 / 2],
                            ]
                        )
                        ** 2
                    ),
                )
            )

    def test_spin_gate(self):
        """test the functionality of the base class for fermionic gates"""
        test_gates = [RLXGate(0.8), RLYGate(2.4), RLZGate(5.6), RLZ2Gate(1.3)]
        with self.subTest("test to_matrix and power"):
            for gate in test_gates:
                exp_matrix = gate.to_matrix() @ gate.to_matrix()
                exp_gate = gate.power(2)
                self.assertTrue(np.allclose(exp_matrix, exp_gate.to_matrix()))

        with self.subTest("test generation of operator"):
            from qiskit.quantum_info import Operator

            for gate in test_gates:
                self.assertTrue(isinstance(Operator(gate), Operator))

    def test_identity_gates(self):
        """test that gates with parameters equal to zero still have a well-defined generator."""
        test_gates = [RLXGate(0.0), RLYGate(0.0), RLZGate(0.0), RLZ2Gate(0.0)]

        for gate in test_gates:
            self.assertIsInstance(gate.generator, SpinOp)
