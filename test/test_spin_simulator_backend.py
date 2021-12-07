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

"""General spin simulator backend tests."""

from time import sleep
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import JobStatus
from qiskit.providers.aer import AerJob
from qiskit.result import Result
from qiskit.test import QiskitTestCase

from qiskit_cold_atom.exceptions import QiskitColdAtomError
from qiskit_cold_atom.spins import SpinSimulator
from qiskit_cold_atom.spins.base_spin_backend import BaseSpinBackend


class TestSpinSimulatorBackend(QiskitTestCase):
    """class to test the FermionSimulatorBackend class."""

    def setUp(self):
        super().setUp()
        self.backend = SpinSimulator()

    def test_initialization(self):
        """test the initialization of the backend"""
        target_config = {
            "backend_name": "spin_simulator",
            "backend_version": "0.0.1",
            "n_qubits": None,
            "basis_gates": None,
            "gates": [],
            "local": True,
            "simulator": True,
            "conditional": False,
            "open_pulse": False,
            "memory": True,
            "max_shots": 1e5,
            "coupling_map": None,
            "description": "a base simulator for spin circuits. Instead of a qubit, each wire "
            "represents a single high-dimensional spin",
        }

        backend = SpinSimulator()
        self.assertIsInstance(backend, BaseSpinBackend)
        self.assertTrue(target_config.items() <= backend.configuration().to_dict().items())

    def test_run_method(self):
        """Test the run method of the backend simulator"""

        with self.subTest("test call"):
            circ = QuantumCircuit(2)
            job = self.backend.run(circ)
            self.assertIsInstance(job, AerJob)
            self.assertIsInstance(job.job_id(), str)
            self.assertIsInstance(job.result(), Result)
            self.assertEqual(job.status(), JobStatus.DONE)

        circ1 = QuantumCircuit(2)
        circ2 = QuantumCircuit(3)

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

        with self.subTest("test dimension of simulation"):
            test_circ = QuantumCircuit(2)
            test_circ.rlx(np.pi / 2, 0)
            test_circ.rly(np.pi / 4, [0, 1])

            statevector_1 = self.backend.run(test_circ, spin=1).result().get_statevector()
            self.assertEqual(len(statevector_1), 3 ** 2)

            statevector_2 = self.backend.run(test_circ, spin=5 / 2).result().get_statevector()
            self.assertEqual(len(statevector_2), 6 ** 2)

        with self.subTest("test irregular spin values"):
            test_circ = QuantumCircuit(2)
            job = self.backend.run(test_circ, spin=5 / 4)
            sleep(0.01)
            self.assertIs(job.status(), JobStatus.ERROR)
            with self.assertRaises(QiskitColdAtomError):
                job.result()

    def test_execute(self):
        """test the ._execute() method internally called by .run()"""

        test_circ = QuantumCircuit(2)
        test_circ.rly(np.pi / 2, 0)
        test_circ.rlx(np.pi / 2, 1)
        test_circ.measure_all()

        result = self.backend.run(test_circ, spin=1, seed=45, shots=5).result()

        with self.subTest("test simulation counts"):
            self.assertEqual(result.get_counts(), {"0 1": 1, "2 2": 1, "1 1": 2, "1 0": 1})

        with self.subTest("test simulation memory"):
            self.assertEqual(result.get_memory(), ["2 2", "1 1", "0 1", "1 0", "1 1"])

        with self.subTest("test simulation statevector"):
            self.assertTrue(
                np.allclose(
                    result.get_statevector(),
                    np.array(
                        [
                            1 / 4,
                            -1j / np.sqrt(8),
                            -1 / 4,
                            1 / np.sqrt(8),
                            -1j / 2,
                            -1 / np.sqrt(8),
                            1 / 4,
                            -1j / np.sqrt(8),
                            -1 / 4,
                        ]
                    ),
                )
            )

        with self.subTest("test simulation unitary"):
            # test the unitary on a single spin-2 example
            test_circ = QuantumCircuit(1)
            test_circ.rlx(np.pi / 2, 0)
            test_circ.rlz(np.pi / 2, 0)
            test_circ.measure_all()

            result = self.backend.run(test_circ, spin=2, seed=45, shots=5).result()

            self.assertTrue(
                np.allclose(
                    result.get_unitary(),
                    np.array(
                        [
                            [-0.25, 0.5j, np.sqrt(3 / 8), -0.5j, -0.25],
                            [-0.5, 0.5j, 0.0, 0.5j, 0.5],
                            [-np.sqrt(3 / 8), 0.0, -0.5, 0.0, -np.sqrt(3 / 8)],
                            [-0.5, -0.5j, 0.0, -0.5j, 0.5],
                            [-0.25, -0.5j, np.sqrt(3 / 8), 0.5j, -0.25],
                        ]
                    ),
                )
            )

        with self.subTest("test time taken"):
            self.assertTrue(result.to_dict()["time_taken"] < 0.5)

        with self.subTest("test result success"):
            self.assertTrue(result.to_dict()["success"])
