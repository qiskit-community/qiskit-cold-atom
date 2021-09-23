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

"""Module for cold-atom spin backends."""

from abc import ABC

from qiskit.providers import BackendV1 as Backend
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_cold_atom import QiskitColdAtomError


class BaseSpinBackend(Backend, ABC):
    """Abstract base class for atomic mixture backends."""

    def get_empty_circuit(self) -> QuantumCircuit:
        """
        Convenience function to set up an empty circuit with the right QuantumRegisters.
        For each atomic species specified in the config file, a quantum register is added to the circuit.

        Returns:
            qc: An empty quantum circuit ready to use in spin-based cold-atom setups.

        Raises:
            QiskitColdAtomError:
                - If backend has no config file.
                - If number of wires of the backend config is not a multiple of the atomic species.
        """
        config = self.configuration().to_dict()

        try:
            num_wires = config["n_qubits"]
            num_species = len(config["atomic_species"])

        except NameError as name_error:
            raise QiskitColdAtomError(
                "backend needs to be initialized with config file first"
            ) from name_error

        if not (isinstance(num_wires, int) and num_wires % num_species == 0):
            raise QiskitColdAtomError(
                "num_wires {num_wires} must be multiple of num_species {num_species}"
            )

        qregs = [
            QuantumRegister(num_wires / num_species, species)
            for species in config["atomic_species"]
        ]

        class_reg = ClassicalRegister(num_wires, "c{}".format(num_wires))
        empty_circuit = QuantumCircuit(*qregs, class_reg)

        return empty_circuit

    def draw(self, qc: QuantumCircuit, **draw_options):
        """Modified circuit drawer to better display atomic mixture quantum circuits.

        For now this method is just an alias to `QuantumCircuit.draw()` but in the future this method
        may be modified and tailored to spin quantum circuits.

        Args:
            qc: The quantum circuit to draw.
            draw_options: Key word arguments for the drawing of circuits.
        """
        qc.draw(**draw_options)
