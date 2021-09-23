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

"""Module for cold-atom fermion backends."""

from abc import ABC
from typing import Union, List, Optional
import numpy as np

from qiskit.providers import BackendV1 as Backend
from qiskit import QuantumCircuit
from qiskit_nature.operators.second_quantization import FermionicOp

from qiskit_cold_atom.fermions.fermion_gate_library import LoadFermions
from qiskit_cold_atom.fermions.fermion_circuit_solver import FermionCircuitSolver
from qiskit_cold_atom.fermions.fermionic_state import FermionicState
from qiskit_cold_atom.exceptions import QiskitColdAtomError


class BaseFermionBackend(Backend, ABC):
    """Abstract base class for fermionic tweezer backends."""

    def initialize_circuit(self, occupations: Union[List[int], List[List[int]]]):
        """
        Initialize a fermionic quantum circuit with the given occupations.

        Args:
            occupations: List of occupation numbers. When ``List[int]`` is given, the occupations
                correspond to the number of indistinguishable fermionic particles in each mode,
                e.g. ``[0, 1, 1, 0]`` implies that sites one and two are occupied by a fermion.
                When ``List[List[int]]`` is given, the occupations describe the number of particles in
                fermionic modes with different (distinguishable) species of fermions. Each
                inner list gives the occupations of one fermionic species.

        Returns:
            circuit: Qiskit QuantumCircuit with a quantum register for each fermionic species
                     initialized with the ``load`` instructions corresponding to the given occupations

        Raises:
            QiskitColdAtomError: If occupations do not match the backend
        """
        try:
            backend_size = self.configuration().to_dict()["n_qubits"]
        except NameError as name_error:
            raise QiskitColdAtomError(
                f"Number of tweezers not specified for {self.name()}"
            ) from name_error

        initial_state = FermionicState(occupations)

        n_wires = initial_state.sites * initial_state.num_species

        if n_wires > backend_size:
            raise QiskitColdAtomError(
                f"{self.name()} supports up to {backend_size} sites, {n_wires} were given"
            )

        # if num_species is specified by the backend, the wires describe different atomic species
        # and the circuit must exactly match the expected wire count of the backend.
        if "num_species" in self.configuration().to_dict().keys():
            num_species = self.configuration().num_species
            if num_species > 1 and n_wires < self.configuration().num_qubits:
                raise QiskitColdAtomError(
                    f"{self.name()} requires circuits with exactly "
                    f"{self.configuration().num_qubits} wires, but an initial occupation of size "
                    f"{n_wires} was given."
                )

        from qiskit.circuit import QuantumRegister

        if initial_state.num_species > 1:
            registers = []
            for i in range(initial_state.num_species):
                registers.append(QuantumRegister(initial_state.sites, f"spin_{i}"))
            circuit = QuantumCircuit(*registers)

        else:
            circuit = QuantumCircuit(QuantumRegister(initial_state.sites, "fer_mode"))

        for i, occupation_list in enumerate(initial_state.occupations):
            for j, occ in enumerate(occupation_list):
                if occ:
                    circuit.append(LoadFermions(), qargs=[i * initial_state.sites + j])

        return circuit

    def measure_observable_expectation(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        observable: FermionicOp,
        shots: int,
        seed: Optional[int] = None,
        num_species: int = 1,
        get_variance: bool = False,
    ):
        """Measure the expectation value of an observable in a state prepared by a given quantum circuit
        that uses fermionic gates. Measurements are added to the entire register if they are not yet
        applied in the circuit.

        Args:
            circuits: QuantumCircuit applying gates with fermionic generators
            observable: A FermionicOp describing an observable of which the expectation value is sampled
            shots: Number of measurement shots taken in case the circuit has measure instructions
            seed: seed for the random number generator of the measurement simulation
            num_species: number of different fermionic species described by the circuits
            get_variance: If True, also returns an estimate of the variance of the observable

        Raises:
            QiskitColdAtomError: if the observable is non-diagonal

        Returns:
            observable_ev: List of the measured expectation values of the observables in given circuits
            variance: List of the estimated variances of of the observables (if get_variance is True)
        """

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        observable_evs = [0] * len(circuits)
        observable_vars = [0] * len(circuits)

        for idx, circuit in enumerate(circuits):

            # check whether the observable is diagonal in the computational basis.
            solver = FermionCircuitSolver(num_species=2)
            solver.preprocess_circuit(circuit)
            observable_mat = solver.operator_to_mat(observable)

            if list(observable_mat.nonzero()[0]) != list(observable_mat.nonzero()[1]):
                raise QiskitColdAtomError(
                    "Measuring general observables that are non-diagonal in the "
                    "computational basis is not yet implemented for "
                    "fermionic backends. This requires non-trivial basis "
                    "transformations that are in general difficult to find and "
                    "depend on the backend's native gate set."
                )

            circuit.remove_final_measurements()
            circuit.measure_all()

            # pylint: disable=unexpected-keyword-arg
            job = self.run(circuit, shots=shots, seed=seed, num_species=num_species)
            counts = job.result().get_counts()

            for bitstring in counts:
                # Extract the index of the measured count-bitstring in the fermionic basis.
                # In contrast to qubits, this is not trivial and requires an additional step.
                ind = solver.basis.get_index_of_measurement(bitstring)

                # contribution to the operator estimate of this outcome
                p = counts[bitstring] / shots
                observable_evs[idx] += p * observable_mat[ind, ind].real

                if get_variance:
                    # contribution to the variance of the operator
                    observable_vars[idx] += (
                        np.sqrt(p * (1 - p) / shots) * observable_mat[ind, ind]
                    ) ** 2

        if get_variance:
            return observable_evs, observable_vars
        else:
            return observable_evs

    def draw(self, qc: QuantumCircuit, **draw_options):
        """Modified circuit drawer to better display atomic mixture quantum  circuits.

        Note that in the future this method may be modified and tailored to fermionic quantum circuits.

        Args:
            qc: The quantum circuit to draw.
            draw_options: Key word arguments for the drawing of circuits.
        """
        qc.draw(**draw_options)
