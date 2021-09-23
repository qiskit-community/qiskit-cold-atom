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

"""tests for circuit_to_cold_atom functions"""

from typing import Dict

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.circuit import Parameter
from qiskit.providers import BackendV1 as Backend
from qiskit.providers.models import BackendConfiguration
from qiskit_cold_atom.exceptions import QiskitColdAtomError
from qiskit_cold_atom.circuit_to_cold_atom import (
    circuit_to_cold_atom,
    circuit_to_data,
    validate_circuits,
)

# These imports are needed to decorate the quantum circuit
import qiskit_cold_atom.spins  # pylint: disable=unused-import
import qiskit_cold_atom.fermions  # pylint: disable=unused-import


class DummyBackend(Backend):
    """dummy backend class for test purposes only"""

    def __init__(self, config_dict: Dict):
        super().__init__(configuration=BackendConfiguration.from_dict(config_dict))

    def run(self, run_input, **options):
        pass

    @classmethod
    def _default_options(cls):
        pass


class TestCircuitToColdAtom(QiskitTestCase):
    """circuit to cold atom tests."""

    def setUp(self):
        super().setUp()
        # Set up a dummy backend from a configuration dictionary

        test_config = {
            "backend_name": "test_backend",
            "backend_version": "0.0.1",
            "simulator": True,
            "local": True,
            "coupling_map": None,
            "description": "dummy backend for testing purposes only",
            "basis_gates": ["hop, int"],
            "memory": False,
            "n_qubits": 5,
            "conditional": False,
            "max_shots": 100,
            "max_experiments": 2,
            "open_pulse": False,
            "gates": [
                {
                    "coupling_map": [[0], [1], [2], [3], [4]],
                    "name": "rLz",
                    "parameters": ["delta"],
                    "qasm_def": "gate rLz(delta) {}",
                },
                {
                    "coupling_map": [[0], [1], [2]],
                    "name": "rLz2",
                    "parameters": ["chi"],
                    "qasm_def": "gate rLz2(chi) {}",
                },
                {
                    "coupling_map": [[0], [1], [2], [3], [4]],
                    "name": "rLx",
                    "parameters": ["omega"],
                    "qasm_def": "gate rx(omega) {}",
                },
            ],
            "supported_instructions": [
                "delay",
                "rLx",
                "rLz",
                "rLz2",
                "measure",
                "barrier",
            ],
        }

        self.dummy_backend = DummyBackend(test_config)

    def test_circuit_to_cold_atom(self):
        """test the circuit_to_cold_atom function"""

        circ1 = QuantumCircuit(3)
        circ1.lx(0.5, [0, 1])
        circ1.lz(0.3, [1, 2])
        circ1.measure_all()

        circ2 = QuantumCircuit(2)
        circ2.lz2(0.5, 1)
        circ2.measure_all()

        shots = 10

        target_output = {
            "experiment_0": {
                "instructions": [
                    ("rLx", [0], [0.5]),
                    ("rLx", [1], [0.5]),
                    ("rLz", [1], [0.3]),
                    ("rLz", [2], [0.3]),
                    ("barrier", [0, 1, 2], []),
                    ("measure", [0], []),
                    ("measure", [1], []),
                    ("measure", [2], []),
                ],
                "num_wires": 3,
                "shots": shots,
            },
            "experiment_1": {
                "instructions": [
                    ("rLz2", [1], [0.5]),
                    ("barrier", [0, 1], []),
                    ("measure", [0], []),
                    ("measure", [1], []),
                ],
                "num_wires": 2,
                "shots": shots,
            },
        }

        actual_output = circuit_to_cold_atom(
            [circ1, circ2], backend=self.dummy_backend, shots=shots
        )

        self.assertEqual(actual_output, target_output)

    def test_validate_circuits(self):
        """test the validation of circuits against the backend configuration"""

        with self.subTest("test size of circuit"):
            circ = QuantumCircuit(6)
            circ.lx(0.4, 2)
            with self.assertRaises(QiskitColdAtomError):
                validate_circuits(circ, backend=self.dummy_backend)

        with self.subTest("test support of native instructions"):
            circ = QuantumCircuit(4)
            # add gate that is not supported by the backend
            circ.hop_fermions([0.5], [0, 1, 2, 3])
            with self.assertRaises(QiskitColdAtomError):
                validate_circuits(circ, backend=self.dummy_backend)

        with self.subTest("check gate coupling map"):
            circ = QuantumCircuit(5)
            circ.lz2(0.5, 4)
            with self.assertRaises(QiskitColdAtomError):
                validate_circuits(circ, backend=self.dummy_backend)

        with self.subTest("test max. allowed circuits"):
            circuits = [QuantumCircuit(2)] * 3
            with self.assertRaises(QiskitColdAtomError):
                circuit_to_cold_atom(circuits=circuits, backend=self.dummy_backend)

        with self.subTest("test max. allowed shots"):
            circuits = QuantumCircuit(2)
            with self.assertRaises(QiskitColdAtomError):
                circuit_to_cold_atom(
                    circuits=circuits, backend=self.dummy_backend, shots=1000
                )

        with self.subTest("test running with unbound parameters"):
            theta = Parameter("θ")
            circ = QuantumCircuit(1)
            circ.lx(theta, 0)
            with self.assertRaises(QiskitColdAtomError):
                validate_circuits(circ, backend=self.dummy_backend)

    def test_circuit_to_data(self):
        """test the circuit to data method"""

        circ = QuantumCircuit(3)
        circ.lx(0.5, [0, 1])
        circ.lz(0.3, [1, 2])
        circ.measure_all()

        target_output = [
            ("rLx", [0], [0.5]),
            ("rLx", [1], [0.5]),
            ("rLz", [1], [0.3]),
            ("rLz", [2], [0.3]),
            ("barrier", [0, 1, 2], []),
            ("measure", [0], []),
            ("measure", [1], []),
            ("measure", [2], []),
        ]

        actual_output = circuit_to_data(circ)

        self.assertEqual(actual_output, target_output)
