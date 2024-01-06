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

"""General fermionic simulator backend tests."""

from time import sleep

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers import JobStatus
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit_aer import AerJob
from qiskit_nature.operators.second_quantization import FermionicOp

from qiskit_cold_atom.exceptions import QiskitColdAtomError
from qiskit_cold_atom.fermions.base_fermion_backend import BaseFermionBackend
from qiskit_cold_atom.fermions.fermion_gate_library import FermionicGate
from qiskit_cold_atom.fermions.fermion_simulator_backend import FermionSimulator


class TestFermionSimulatorBackend(QiskitTestCase):
    """class to test the FermionSimulatorBackend class."""

    def setUp(self):
        super().setUp()
        self.backend = FermionSimulator()

    def test_initialization(self):
        """test the initialization of the backend"""
        target_config = {
            "backend_name": "fermion_simulator",
            "backend_version": "0.0.1",
            "n_qubits": 20,
            "basis_gates": None,
            "gates": [],
            "local": False,
            "simulator": True,
            "conditional": False,
            "open_pulse": False,
            "memory": True,
            "max_shots": 1e5,
            "coupling_map": None,
            "description": r"a base simulator for fermionic circuits. Instead of qubits, "
            r"each wire represents a single fermionic mode",
        }

        backend = FermionSimulator()
        self.assertIsInstance(backend, BaseFermionBackend)
        self.assertTrue(target_config.items() <= backend.configuration().to_dict().items())

    def test_run_method(self):
        """Test the run method of the backend simulator"""

        with self.subTest("test call"):
            circ = self.backend.initialize_circuit([[0, 1], [1, 0]])
            job = self.backend.run(circ)
            self.assertIsInstance(job, AerJob)
            self.assertIsInstance(job.job_id(), str)
            self.assertIsInstance(job.result(), Result)
            sleep(0.01)
            self.assertEqual(job.status(), JobStatus.DONE)

        circ1 = self.backend.initialize_circuit([[0, 1], [1, 0]])
        circ2 = self.backend.initialize_circuit([[1, 1], [1, 0]])

        with self.subTest("test call with multiple circuits"):
            job = self.backend.run([circ1, circ2])
            self.assertIsInstance(job, AerJob)

        with self.subTest("test shot number"):
            target_shots = 123
            job = self.backend.run([circ1, circ2], shots=target_shots)
            for exp in job.result().results:
                self.assertEqual(exp.shots, target_shots)

        with self.subTest("test seed of RNG"):
            target_seed = 123
            job = self.backend.run([circ1, circ2], seed=target_seed)
            for exp in job.result().results:
                self.assertEqual(exp.header.random_seed, target_seed)

        with self.subTest("test number of fermionic species"):
            # define a circuit that conserves the particle number per fermionic spin species
            test_circ = QuantumCircuit(4)
            test_circ.fload([0, 3])
            test_circ.fhop([np.pi / 4], [0, 1, 2, 3])

            statevector_1 = self.backend.run(test_circ).result().get_statevector()
            self.assertEqual(len(statevector_1), 6)
            # check whether specifying the number of species reduces the dimension of the simulation
            statevector_2 = self.backend.run(test_circ, num_species=2).result().get_statevector()
            self.assertEqual(len(statevector_2), 4)

    def test_execute(self):
        """test the ._execute() method internally called by .run()"""

        with self.subTest("test partial measurement"):
            circ_meas = QuantumCircuit(2, 2)
            circ_meas.fload(0)
            circ_meas.measure(0, 0)
            with self.assertWarns(UserWarning):
                self.backend.run(circ_meas)

        test_circ = QuantumCircuit(4)
        test_circ.fload([0, 3])
        test_circ.fhop([np.pi / 4], [0, 1, 2, 3])
        test_circ.fint(np.pi, [0, 1, 2, 3])
        test_circ.measure_all()

        result = self.backend.run(test_circ, num_species=2, seed=40, shots=5).result()

        with self.subTest("test simulation counts"):
            self.assertEqual(result.get_counts(), {"1010": 1, "0110": 3, "0101": 1})

        with self.subTest("test simulation memory"):
            self.assertEqual(result.get_memory(), ["0110", "0101", "1010", "0110", "0110"])

        with self.subTest("test simulation statevector"):
            self.assertTrue(
                np.allclose(result.get_statevector(), np.array([-0.5j, -0.5, 0.5, -0.5j]))
            )

        with self.subTest("test simulation unitary"):
            self.assertTrue(
                np.allclose(
                    result.get_unitary(),
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

        with self.subTest("test time taken"):
            self.assertTrue(result.to_dict()["time_taken"] < 0.1)

        with self.subTest("test result success"):
            self.assertTrue(result.to_dict()["success"])

    def test_initialize_circuit(self):
        """test of initialize_circuit inherited from the abstract base class BaseFermionBackend"""

        with self.subTest("Initialize circuit with single species of fermions"):
            actual_circ = self.backend.initialize_circuit([0, 1, 0, 1])
            target_circ = QuantumCircuit(QuantumRegister(4, "fer_mode"))
            target_circ.fload(1)
            target_circ.fload(3)
            self.assertEqual(actual_circ, target_circ)

        with self.subTest("Initialize circuit with multiple species of fermions"):
            actual_circ = self.backend.initialize_circuit([[0, 1], [0, 1]])
            target_circ = QuantumCircuit(QuantumRegister(2, "spin_0"), QuantumRegister(2, "spin_1"))
            target_circ.fload(1)
            target_circ.fload(3)
            self.assertEqual(actual_circ, target_circ)

        with self.subTest("check maximum size of circuit"):
            with self.assertRaises(QiskitColdAtomError):
                self.backend.initialize_circuit(np.ones(30, dtype=int).tolist())

    def test_measure_observable_expectation(self):
        """test of the measure_observable_expectation method inherited from the abstract base class
        BaseFermionBackend"""

        with self.subTest("test error for non-diagonal observables"):
            non_diag_observable = FermionicOp("+-NI")
            test_circ = self.backend.initialize_circuit([0, 1, 0, 1])
            with self.assertRaises(QiskitColdAtomError):
                self.backend.measure_observable_expectation(
                    test_circ, non_diag_observable, shots=10
                )

        with self.subTest("test match of dimensionality"):
            observable_too_small = FermionicOp("NI")
            test_circ = self.backend.initialize_circuit([0, 1, 0, 1])
            with self.assertRaises(QiskitColdAtomError):
                self.backend.measure_observable_expectation(
                    test_circ, observable_too_small, shots=10
                )

        with self.subTest("test single measurement circuit"):
            observable_1 = FermionicOp("INEI")
            observable_2 = FermionicOp("INII") + FermionicOp("IIEI")
            observable_3 = FermionicOp("NNII")

            eval_1 = self.backend.measure_observable_expectation(
                circuits=self.backend.initialize_circuit([0, 1, 0, 1]),
                observable=observable_1,
                shots=1,
            )
            eval_2 = self.backend.measure_observable_expectation(
                circuits=self.backend.initialize_circuit([0, 1, 0, 1]),
                observable=observable_2,
                shots=1,
            )
            eval_3 = self.backend.measure_observable_expectation(
                circuits=self.backend.initialize_circuit([0, 1, 0, 1]),
                observable=observable_3,
                shots=1,
            )

            self.assertEqual(eval_1, [1.0])
            self.assertEqual(eval_2, [2.0])
            self.assertEqual(eval_3, [0.0])

        with self.subTest("test multiple measurement circuits"):
            test_circ_1 = self.backend.initialize_circuit([0, 1, 0, 1])
            test_circ_2 = self.backend.initialize_circuit([1, 0, 0, 1])
            observable = FermionicOp("INEI")
            expval = self.backend.measure_observable_expectation(
                [test_circ_1, test_circ_2], observable, shots=1
            )
            self.assertEqual(expval, [1.0, 0.0])

    def test_parameterized_circuits(self):
        """Test that parameterized circuits work."""
        from qiskit.circuit import Parameter

        theta = Parameter("theta")

        test_circ = QuantumCircuit(4)
        test_circ.fload([0, 3])
        test_circ.fhop([theta], [0, 1, 2, 3])

        with self.subTest("test running with unbound parameters:"):
            with self.assertRaises(TypeError):
                self.assertTrue(isinstance(self.backend.run(test_circ).result(), Result))

        with self.subTest("test running with bound parameters"):
            bound_circ = test_circ.bind_parameters([0.2])
            self.assertTrue(isinstance(self.backend.run(bound_circ).result(), Result))

    def test_permutation_invariance(self):
        """Test that a permutation-invariant gate doesn't care about qubit order."""
        generator = FermionicOp(
            [("+_0 -_1", 1), ("+_1 -_0", 1)],
            register_length=2,
        )
        gate = FermionicGate(name="test", num_modes=2, generator=generator)

        circuit01 = self.backend.initialize_circuit([1, 0])
        circuit01.append(gate, [0, 1])
        vec01 = self.backend.run(circuit01).result().get_statevector()

        circuit10 = self.backend.initialize_circuit([1, 0])
        circuit10.append(gate, [1, 0])
        vec10 = self.backend.run(circuit10).result().get_statevector()

        np.testing.assert_allclose(vec01, vec10)
