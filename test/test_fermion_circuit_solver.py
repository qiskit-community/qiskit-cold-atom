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

"""Fermionic circuit solver tests."""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit_nature.second_q.operators import FermionicOp, SpinOp

from qiskit_cold_atom.fermions.fermion_circuit_solver import (
    FermionCircuitSolver,
    FermionicBasis,
)
from qiskit_cold_atom.fermions.fermion_gate_library import Hop
from qiskit_cold_atom.exceptions import QiskitColdAtomError

# Black import needed to decorate the quantum circuit with spin gates.
import qiskit_cold_atom.spins  # pylint: disable=unused-import


class TestFermionCircuitSolver(QiskitTestCase):
    """class to test the FermionCircuitSolver class."""

    def setUp(self):
        super().setUp()
        # Setup two solvers with different number of fermionic species
        self.solver1 = FermionCircuitSolver()
        self.solver2 = FermionCircuitSolver(num_species=2)

    def test_basis_setter(self):
        """test max. dimension of the basis"""
        self.solver1.max_dimension = 500
        with self.assertRaises(QiskitColdAtomError):
            self.solver1.basis = FermionicBasis(sites=12, n_particles=6)

    def test_preprocess_circuit(self):
        """test the preprocessing of the circuit"""
        circ = QuantumCircuit(4, 4)
        circ.fload([0, 2])
        circ.fhop([0.5], [0, 1, 2, 3])
        with self.subTest("spin conserving circuit"):
            self.solver2.preprocess_circuit(circ)
            self.assertEqual(self.solver2._dim, 4)
        with self.subTest("non spin conserving circuit"):
            self.solver1.preprocess_circuit(circ)
            self.assertEqual(self.solver1._dim, 6)

    def test_get_initial_state(self):
        """test initialization of the state for the simulation"""
        circ = QuantumCircuit(4, 4)
        circ.fload([0, 3])
        self.solver2.preprocess_circuit(circ)
        init_state = self.solver2.get_initial_state(circ)
        target = np.array([0, 0, 1, 0])
        self.assertTrue(np.all(init_state.toarray().T == target))

    def test_embed_operator(self):
        """test embedding of an operator"""
        fer_op = FermionicOp("+-")
        spin_op = SpinOp("+-")
        num_wires = 4
        qargs = [1, 3]
        qargs_wrong = [0, 1, 3]

        with self.subTest("check operator type"):
            with self.assertRaises(QiskitColdAtomError):
                self.solver1._embed_operator(spin_op, num_wires, qargs)

        with self.subTest("check operator wiring"):
            with self.assertRaises(QiskitColdAtomError):
                self.solver1._embed_operator(fer_op, num_wires, qargs_wrong)

        with self.subTest("operator embedding"):
            embedded_op = self.solver1._embed_operator(fer_op, num_wires, qargs)
            target_op = FermionicOp("+_1 -_3", register_length=4)
            self.assertTrue(
                set(embedded_op.simplify().to_list(display_format="dense"))
                == set(target_op.simplify().to_list(display_format="dense"))
            )

    def test_conservation_checks(self):
        """test the checks for conservation of spin-species."""
        with self.subTest("check operator type"):
            circ = QuantumCircuit(4, 4)
            circ.fhop([0.5], [0, 1, 2, 3])
            circ.rlx(0.5, 0)  # apply gate with a SpinOp generator
            with self.assertRaises(QiskitColdAtomError):
                self.solver1._check_conservations(circ)

        with self.subTest("check compatibility with number of species"):
            circ = QuantumCircuit(5, 5)
            circ.fhop([0.5], [0, 1, 2, 3])
            self.assertTrue(self.solver1._check_conservations(circ) == (True, True))
            with self.assertRaises(QiskitColdAtomError):
                self.solver2._check_conservations(circ)

        with self.subTest("spin conserved"):
            circ = QuantumCircuit(4, 4)
            circ.fload([0, 3])
            circ.fhop([0.5], [0, 1, 2, 3])
            self.assertTrue(self.solver2._check_conservations(circ) == (True, True))

        with self.subTest("spin not conserved"):
            circ = QuantumCircuit(4, 4)
            circ.fload([0, 3])
            circ.fhop([0.5], [0, 1, 2, 3])
            circ.frx(0.3, [0, 2])  # non spin-conserving gate
            self.assertTrue(self.solver2._check_conservations(circ) == (True, False))

    def test_operator_to_mat(self):
        """test matrix representation of fermionic gates"""

        with self.subTest("check operator type"):
            spin_op = SpinOp("+-")
            with self.assertRaises(QiskitColdAtomError):
                self.solver1.operator_to_mat(spin_op)

        circ = QuantumCircuit(4, 4)
        circ.fload([0, 3])
        circ.fhop([0.5], [0, 1, 2, 3])

        with self.subTest("check dimensionality of operator"):
            self.solver2.preprocess_circuit(circ)
            fer_op_wrong = FermionicOp("+-I")
            fer_op_correct = FermionicOp("+-II", register_length=4)
            with self.assertRaises(QiskitColdAtomError):
                self.solver2.operator_to_mat(fer_op_wrong)
            self.solver2.operator_to_mat(fer_op_correct)

        with self.subTest("test matrix representation"):
            self.solver2.preprocess_circuit(circ)
            target = np.array(
                [
                    [0.0, -0.5, -0.5, 0.0],
                    [-0.5, 0.0, 0.0, -0.5],
                    [-0.5, 0.0, 0.0, -0.5],
                    [0.0, -0.5, -0.5, 0.0],
                ]
            )
            test_op = self.solver2.operator_to_mat(Hop(num_modes=4, j=[0.5]).generator)
            self.assertTrue(np.all(test_op.toarray() == target))

    def test_draw_shots(self):
        """test drawing of the shots from a measurement distribution"""
        circ = QuantumCircuit(4, 4)
        circ.fload([0, 3])
        circ.fhop([0.5], [0, 1, 2, 3])
        self.solver2.preprocess_circuit(circ)

        with self.subTest("check missing shot number"):
            # error because the number of shots is not specified
            with self.assertRaises(QiskitColdAtomError):
                self.solver2.draw_shots(np.ones(4) / 4)

        self.solver2.shots = 5

        with self.subTest("check match of dimensions"):
            # error because there is a mismatch in the dimension
            with self.assertRaises(QiskitColdAtomError):
                self.solver2.draw_shots(np.ones(3) / 3)

        with self.subTest("formatting of measurement outcomes"):
            self.solver2.seed = 40
            outcomes = self.solver2.draw_shots(np.ones(4) / 4)
            self.assertEqual(outcomes, ["0110", "0101", "1010", "0110", "0110"])

    def test_to_operators(self):
        """test the to_operators method inherited form BaseCircuitSolver"""

        test_circ = QuantumCircuit(4, 4)
        test_circ.fload([0, 3])
        test_circ.fhop([0.5], [0, 1, 2, 3])
        test_circ.fint(1.0, [0, 1, 2, 3])
        test_circ.measure_all()

        with self.subTest("test ignore barriers"):
            self.solver1.ignore_barriers = False
            with self.assertRaises(NotImplementedError):
                self.solver1.to_operators(test_circ)
            self.solver1.ignore_barriers = True

        with self.subTest("check for gate generators"):
            qubit_circ = QuantumCircuit(1)
            qubit_circ.h(0)
            with self.assertRaises(QiskitColdAtomError):
                self.solver1.to_operators(qubit_circ)

        with self.subTest("gate after previous measurement instruction"):
            meas_circ = QuantumCircuit(4, 4)
            meas_circ.measure_all()
            meas_circ.fhop([0.5], [0, 1, 2, 3])
            with self.assertRaises(QiskitColdAtomError):
                self.solver1.to_operators(meas_circ)

        with self.subTest("check returned operators"):
            operators = self.solver1.to_operators(test_circ)
            target = [
                FermionicOp(
                    [
                        ("+-II", -0.5),
                        ("-+II", 0.5),
                        ("II+-", -0.5),
                        ("II-+", 0.5),
                    ]
                ),
                FermionicOp([("NINI", 1), ("ININ", 1)]),
            ]
            for i, op in enumerate(operators):
                self.assertEqual(
                    set(op.simplify().to_list(display_format="dense")),
                    set(target[i].simplify().to_list(display_format="dense")),
                )

    def test_call_method(self):
        """test the call method inherited form BaseCircuitSolver that simulates a circuit"""

        test_circ = QuantumCircuit(4)
        test_circ.fload([0, 3])
        test_circ.fhop([np.pi / 4], [0, 1, 2, 3])
        test_circ.fint(np.pi, [0, 1, 2, 3])

        with self.subTest("running the circuit"):
            self.solver2.shots = 5
            self.solver2.seed = 40
            simulation = self.solver2(test_circ)

            self.assertEqual(simulation["memory"], ["0110", "0101", "1010", "0110", "0110"])
            self.assertEqual(simulation["counts"], {"1010": 1, "0110": 3, "0101": 1})
            self.assertTrue(
                np.allclose(simulation["statevector"], np.array([-0.5j, -0.5, 0.5, -0.5j]))
            )
            self.assertTrue(
                np.allclose(
                    simulation["unitary"],
                    np.array(
                        [
                            [-0.5, -0.5j, -0.5j, 0.5],
                            [0.5j, 0.5, -0.5, 0.5j],
                            [0.5j, -0.5, 0.5, 0.5j],
                            [0.5, -0.5j, -0.5j, -0.5],
                        ]
                    ),
                )
            )

        with self.subTest("check for maximum dimension"):
            self.solver2.max_dimension = 3
            with self.assertRaises(QiskitColdAtomError):
                simulation = self.solver2(test_circ)
            self.solver2.max_dimension = 100

        with self.subTest("check if shots are specified"):
            self.solver2.shots = None
            simulation = self.solver2(test_circ)
            self.assertEqual(simulation["memory"], [])
            self.assertEqual(simulation["counts"], {})
