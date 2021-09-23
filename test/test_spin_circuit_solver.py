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

"""Spin circuit solver tests"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit_nature.operators.second_quantization import FermionicOp, SpinOp
from qiskit_cold_atom.spins.spin_circuit_solver import SpinCircuitSolver
from qiskit_cold_atom.exceptions import QiskitColdAtomError


class TestSpinCircuitSolver(QiskitTestCase):
    """class to test the SpinCircuitSolver class."""

    def setUp(self):
        super().setUp()
        # Setup the simulator
        self.solver = SpinCircuitSolver(spin=3 / 2)

    def test_spin_solver_initialization(self):
        """test constructor of SpinCircuitSolver"""
        with self.assertRaises(QiskitColdAtomError):
            SpinCircuitSolver(spin=2 / 3)

    def test_get_initial_state(self):
        """test initialization of the state for the simulation"""
        circ = QuantumCircuit(1)
        init_state = self.solver.get_initial_state(circ)
        target = np.array([1, 0, 0, 0])
        self.assertTrue(np.alltrue(init_state.toarray().T == target))

    def test_embed_operator(self):
        """test embedding of an operator"""
        fer_op = FermionicOp("+-")
        spin_op = SpinOp("+-")
        num_wires = 4
        qargs = [1, 3]
        qargs_wrong = [0, 1, 3]

        with self.subTest("check operator type"):
            with self.assertRaises(QiskitColdAtomError):
                self.solver._embed_operator(fer_op, num_wires, qargs)

        with self.subTest("check operator wiring"):
            with self.assertRaises(QiskitColdAtomError):
                self.solver._embed_operator(spin_op, num_wires, qargs_wrong)

        with self.subTest("operator embedding"):
            embedded_op = self.solver._embed_operator(spin_op, num_wires, qargs)
            target_op = SpinOp("+_1 -_3", spin=3 / 2, register_length=4)
            self.assertTrue(
                set(embedded_op.reduce().to_list()) == set(target_op.reduce().to_list())
            )

    def test_preprocess_circuit(self):
        """test whether preprocessing of the circuit correctly sets the dimension"""
        circ = QuantumCircuit(2)
        self.solver.preprocess_circuit(circ)
        self.assertEqual(self.solver.dim, 4 ** 2)

    def test_draw_shots(self):
        """test drawing of the shots from a measurement distribution"""
        circ = QuantumCircuit(2)
        self.solver.preprocess_circuit(circ)

        with self.subTest("check missing shot number"):
            # error because the number of shots is not specified
            with self.assertRaises(QiskitColdAtomError):
                self.solver.draw_shots(np.ones(16) / 16)

        self.solver.shots = 5

        with self.subTest("check match of dimensions"):
            # error because there is a mismatch in the dimension
            with self.assertRaises(QiskitColdAtomError):
                self.solver.draw_shots(np.ones(15) / 15)

        with self.subTest("formatting of measurement outcomes"):

            self.solver.seed = 45
            outcomes = self.solver.draw_shots(np.ones(16) / 16)
            self.assertEqual(outcomes, ["3 3", "0 2", "0 1", "1 0", "3 1"])

    def test_to_operators(self):
        """test the to_operators method inherited form BaseCircuitSolver"""

        test_circ = QuantumCircuit(2)
        test_circ.lx(0.5, [0, 1])
        test_circ.lz2(0.25, 1)
        test_circ.measure_all()

        with self.subTest("test ignore barriers"):
            self.solver.ignore_barriers = False
            with self.assertRaises(NotImplementedError):
                self.solver.to_operators(test_circ)
            self.solver.ignore_barriers = True

        with self.subTest("check for gate generators"):
            qubit_circ = QuantumCircuit(1)
            qubit_circ.h(0)
            with self.assertRaises(QiskitColdAtomError):
                self.solver.to_operators(qubit_circ)

        with self.subTest("gate after previous measurement instruction"):
            meas_circ = QuantumCircuit(2)
            meas_circ.measure_all()
            meas_circ.lx(0.5, 0)
            with self.assertRaises(QiskitColdAtomError):
                self.solver.to_operators(meas_circ)

        with self.subTest("check returned operators"):
            operators = self.solver.to_operators(test_circ)
            target = [
                SpinOp([("X_0", (0.5 + 0j))], spin=3 / 2, register_length=2),
                SpinOp([("X_1", (0.5 + 0j))], spin=3 / 2, register_length=2),
                SpinOp([("Z_1^2", (0.25 + 0j))], spin=3 / 2, register_length=2),
            ]

            for i, op in enumerate(operators):
                self.assertEqual(
                    set(op.reduce().to_list()), set(target[i].reduce().to_list())
                )

    def test_call_method(self):
        """test the call method inherited form BaseCircuitSolver that simulates a circuit"""

        test_circ = QuantumCircuit(1)
        test_circ.lx(np.pi / 2, 0)
        test_circ.measure_all()

        with self.subTest("running the circuit"):
            self.solver.shots = 5
            self.solver.seed = 45
            simulation = self.solver(test_circ)

            self.assertEqual(simulation["memory"], ["3", "2", "1", "0", "1"])
            self.assertEqual(simulation["counts"], {"0": 1, "1": 2, "2": 1, "3": 1})
            self.assertTrue(
                np.allclose(
                    simulation["statevector"],
                    np.array(
                        [
                            np.sqrt(1 / 8),
                            -1j * np.sqrt(3 / 8),
                            -np.sqrt(3 / 8),
                            1j * np.sqrt(1 / 8),
                        ]
                    ),
                )
            )
            self.assertTrue(
                np.allclose(
                    simulation["unitary"],
                    np.array(
                        [
                            [
                                np.sqrt(1 / 8),
                                -1j * np.sqrt(3 / 8),
                                -np.sqrt(3 / 8),
                                1j * np.sqrt(1 / 8),
                            ],
                            [
                                -1j * np.sqrt(3 / 8),
                                -np.sqrt(1 / 8),
                                -1j * np.sqrt(1 / 8),
                                -np.sqrt(3 / 8),
                            ],
                            [
                                -np.sqrt(3 / 8),
                                -1j * np.sqrt(1 / 8),
                                -np.sqrt(1 / 8),
                                -1j * np.sqrt(3 / 8),
                            ],
                            [
                                1j * np.sqrt(1 / 8),
                                -np.sqrt(3 / 8),
                                -1j * np.sqrt(3 / 8),
                                np.sqrt(1 / 8),
                            ],
                        ]
                    ),
                )
            )

        with self.subTest("check for maximum dimension"):
            self.solver.max_dimension = 3
            with self.assertRaises(QiskitColdAtomError):
                self.solver(test_circ)
            self.solver.max_dimension = 100

        with self.subTest("check if shots are specified"):
            self.solver.shots = None
            simulation = self.solver(test_circ)
            self.assertEqual(simulation["memory"], [])
            self.assertEqual(simulation["counts"], {})
            self.solver.shots = 5

        multiple_wire_circ = QuantumCircuit(2)
        multiple_wire_circ.lx(np.pi / 2, [0])
        multiple_wire_circ.lz(-np.pi / 2, [0, 1])

        with self.subTest("formatting of multiple wires"):
            self.solver.seed = 45
            simulation = self.solver(multiple_wire_circ)
            self.assertTrue(simulation["memory"], ["0 3", "0 2", "0 1", "0 0", "0 1"])
            self.assertTrue(
                simulation["counts"], {"0 2": 1, "0 0": 1, "0 3": 1, "0 1": 2}
            )

        with self.subTest("check equivalence to qubits for spin-1/2"):
            from qiskit import Aer

            qubit_circ = QuantumCircuit(2)
            qubit_circ.rx(np.pi / 2, [0])
            qubit_circ.rz(-np.pi / 2, [0, 1])
            qubit_circ.save_unitary()
            qubit_backend = Aer.get_backend("aer_simulator")
            job = qubit_backend.run(qubit_circ)
            qubit_unitary = job.result().get_unitary()

            spin_half_solver = SpinCircuitSolver(spin=1 / 2)
            simulation = spin_half_solver(multiple_wire_circ)
            spin_unitary = simulation["unitary"]
            # Switch some axes in the spin simulator unitary because the basis of qiskit_nature.SpinOp
            # uses an ordering of the states (00, 10, 01, 11) that is different from qiskit Aer which
            # uses (00, 01, 10, 11)
            spin_unitary[[1, 2]] = spin_unitary[[2, 1]]
            spin_unitary[:, [1, 2]] = spin_unitary[:, [2, 1]]

            self.assertTrue(np.allclose(qubit_unitary, spin_unitary))
