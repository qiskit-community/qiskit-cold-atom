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

""" Fermionic gate tests """

import numpy as np
from scipy.linalg import expm
from qiskit.test import QiskitTestCase
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_cold_atom.fermions.fermion_simulator_backend import FermionSimulator
from qiskit_cold_atom.fermions.fermion_gate_library import (
    Hopping,
    Interaction,
    LocalPhase,
    FermionRX,
    FermionRY,
    FermionRZ,
    FermiHubbard,
)


class TestFermionicGates(QiskitTestCase):
    """Tests for the fermionic gates"""

    def setUp(self):
        super().setUp()
        self.backend = FermionSimulator()

    def test_interaction_gate(self):
        """check matrix form of interaction gate in a two-tweezer example"""
        u_val = np.pi / 2
        circ = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ.append(Interaction(4, u_val), qargs=[0, 1, 2, 3])
        # add gate to circuit via the @add_gate-decorated method
        circ_decorated = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ_decorated.int_fermions(u_val, [0, 1, 2, 3])

        for circuit in [circ, circ_decorated]:
            unitary = self.backend.run(circuit, num_species=2).result().get_unitary()
            self.assertTrue(
                np.allclose(
                    unitary,
                    expm(
                        -1j
                        * np.array(
                            [
                                [u_val, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, u_val],
                            ]
                        )
                    ),
                )
            )

    def test_hopping_gate(self):
        """check matrix form of hopping gate in a two-tweezer example"""
        j = np.pi / 4
        circ = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ.append(Hopping(4, [j]), qargs=[0, 1, 2, 3])
        # add gate to circuit via the @add_gate-decorated method
        circ_decorated = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ_decorated.hop_fermions([j], [0, 1, 2, 3])

        for circuit in [circ, circ_decorated]:
            unitary = self.backend.run(circuit, num_species=2).result().get_unitary()
            self.assertTrue(
                np.allclose(
                    unitary,
                    expm(
                        -1j
                        * np.array(
                            [
                                [0, -j, -j, 0],
                                [-j, 0, 0, -j],
                                [-j, 0, 0, -j],
                                [0, -j, -j, 0],
                            ]
                        )
                    ),
                )
            )

    def test_phase_gate(self):
        """check matrix form of phase gate in a two-tweezer example"""
        phi1, phi2 = np.pi / 4, np.pi / 8
        circ = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ.append(LocalPhase(4, [phi1, phi2]), qargs=[0, 1, 2, 3])
        # add gate to circuit via the @add_gate-decorated method
        circ_decorated = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ_decorated.phase_fermions([phi1, phi2], [0, 1, 2, 3])

        for circuit in [circ, circ_decorated]:
            unitary = self.backend.run(circuit, num_species=2).result().get_unitary()

            self.assertTrue(
                np.allclose(
                    unitary,
                    expm(
                        -1j
                        * np.array(
                            [
                                [2 * phi2, 0, 0, 0],
                                [0, phi2 + phi1, 0, 0],
                                [0, 0, phi2 + phi1, 0],
                                [0, 0, 0, 2 * phi1],
                            ]
                        )
                    ),
                )
            )

    def test_spin_rx_gate(self):
        """check matrix form of spin_rx gate in a two-tweezer example"""
        phi = np.pi / 4
        circ = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ.append(FermionRX(phi), qargs=[0, 2])
        # add gate to circuit via the @add_gate-decorated method
        circ_decorated = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ_decorated.rx_fermions(phi, [0, 2])

        for circuit in [circ, circ_decorated]:
            unitary = self.backend.run(circuit, num_species=2).result().get_unitary()

            self.assertTrue(
                np.allclose(
                    unitary,
                    expm(
                        -1j
                        * np.array(
                            [
                                [0, 0, 0, phi, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, -phi],
                                [phi, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, -phi, 0, 0, 0],
                            ]
                        )
                    ),
                )
            )

    def test_spin_ry_gate(self):
        """check matrix form of spin_ry gate in a two-tweezer example"""
        phi = np.pi / 4
        circ = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ.append(FermionRY(phi), qargs=[0, 2])
        # add gate to circuit via the @add_gate-decorated method
        circ_decorated = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ_decorated.ry_fermions(phi, [0, 2])

        for circuit in [circ, circ_decorated]:
            unitary = self.backend.run(circuit, num_species=2).result().get_unitary()

            self.assertTrue(
                np.allclose(
                    unitary,
                    expm(
                        -1j
                        * np.array(
                            [
                                [0, 0, 0, 1j * phi, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, -1j * phi],
                                [-1j * phi, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 1j * phi, 0, 0, 0],
                            ]
                        )
                    ),
                )
            )

    def test_spin_rz_gate(self):
        """check matrix form of spin_rz gate in a two-tweezer example"""
        phi = np.pi / 4
        circ = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ.append(FermionRZ(phi), qargs=[0, 2])
        # add gate to circuit via the @add_gate-decorated method
        circ_decorated = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ_decorated.rz_fermions(phi, [0, 2])

        for circuit in [circ, circ_decorated]:
            unitary = self.backend.run(circuit, num_species=2).result().get_unitary()

            self.assertTrue(
                np.allclose(
                    unitary,
                    expm(
                        -1j
                        * np.array(
                            [
                                [0, 0, 0, 0],
                                [0, -phi, 0, 0],
                                [0, 0, phi, 0],
                                [0, 0, 0, 0],
                            ]
                        )
                    ),
                )
            )

    def test_fermionic_gate_class(self):
        """test the functionality of the base class for fermionic gates"""

        test_gates = [
            Hopping(num_modes=4, j=[0.5]),
            Interaction(num_modes=8, u=2.0),
            LocalPhase(num_modes=2, mu=[1.0]),
            FermionRX(phi=0.5),
            FermionRY(phi=-0.7),
            FermionRZ(phi=1.3),
            FermiHubbard(num_modes=4, j=[0.5], u=1.0, mu=[0.4, 1.2]),
        ]

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
        test_gates = [
            Hopping(num_modes=4, j=[0.0]),
            Interaction(num_modes=4, u=0.0),
            LocalPhase(num_modes=2, mu=[0.0]),
            FermionRX(phi=0.0),
            FermionRY(phi=-0.0),
            FermionRZ(phi=0.0),
            FermiHubbard(num_modes=4, j=[0.0], u=0.0, mu=[0.0, 0.0]),
        ]

        for gate in test_gates:
            self.assertIsInstance(gate.generator, FermionicOp)
