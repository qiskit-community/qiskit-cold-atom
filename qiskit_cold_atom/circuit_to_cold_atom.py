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

"""Module to convert cold atom circuits to dictionaries."""

from typing import List, Union, Optional

from qiskit import QuantumCircuit
from qiskit.providers import BackendV1 as Backend
from qiskit_cold_atom.exceptions import QiskitColdAtomError


def validate_circuits(
    circuits: Union[List[QuantumCircuit], QuantumCircuit],
    backend: Backend,
    shots: Optional[int] = None,
) -> None:
    """
    Performs validity checks on circuits against the configuration of the backends. This checks whether
    all applied instructions in the circuit are accepted by the backend and whether the applied gates
    comply with their respective coupling maps.

    Args:
        circuits: The circuits that need to be run.
        backend: The backend on which the circuit should be run.
        shots: The number of shots for each circuit.

    Raises:
        QiskitColdAtomError: - If the maximum number of experiments or shots specified by the backend is
                             exceeded
                             - If the backend does not support an instruction given in the circuit
                             - If the width of the circuit is too large
                             - If the circuit has unbound parameters
    """

    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    # check for number of experiments allowed by the backend
    if backend.configuration().max_experiments:
        max_circuits = backend.configuration().max_experiments
        if len(circuits) > max_circuits:
            raise QiskitColdAtomError(
                f"{backend.name()} allows for max. {max_circuits} different circuits; "
                f"but {len(circuits)} circuits were given"
            )
    # check for number of individual shots allowed by the backend
    if backend.configuration().max_shots and shots:
        max_shots = backend.configuration().max_shots
        if shots > max_shots:
            raise QiskitColdAtomError(
                f"{backend.name()} allows for max. {max_shots} shots per circuit; "
                f"{shots} shots were requested"
            )

    for circuit in circuits:

        try:
            native_gates = {gate.name: gate.coupling_map for gate in backend.configuration().gates}
            native_instructions = backend.configuration().supported_instructions
        except NameError as name_error:
            raise QiskitColdAtomError(
                "backend needs to be initialized with config file first"
            ) from name_error

        if circuit.num_qubits > backend.configuration().num_qubits:
            raise QiskitColdAtomError(
                f"{backend.name()} supports circuits with up to "
                f"{backend.configuration().num_qubits} wires, but"
                f"{circuit.num_qubits} wires were given."
            )

        # If num_species is specified by the backend, the wires describe different atomic species
        # and the circuit must exactly match the expected wire count of the backend.
        if "num_species" in backend.configuration().to_dict().keys():
            num_species = backend.configuration().num_species
            if num_species > 1 and circuit.num_qubits < backend.configuration().num_qubits:
                raise QiskitColdAtomError(
                    f"{backend.name()} requires circuits to be submitted with exactly "
                    f"{backend.configuration().num_qubits} wires, but "
                    f"{circuit.num_qubits} wires were given."
                )

        for inst in circuit.data:
            # get the correct wire indices of the instruction with respect
            # to the total index of the qubit objects in the circuit
            wires = [circuit.qubits.index(qubit) for qubit in inst[1]]

            for param in inst[0].params:
                try:
                    float(param)
                except TypeError as type_error:
                    raise QiskitColdAtomError(
                        "Cannot run circuit with unbound parameters."
                    ) from type_error

            # check if instruction is supported by the backend
            name = inst[0].name
            if name not in native_instructions:
                raise QiskitColdAtomError(f"{backend.name()} does not support {name}")

            # for the gates, check whether coupling map fits
            if name in native_gates:
                couplings = native_gates[name]
                if wires not in couplings:
                    raise QiskitColdAtomError(
                        f"coupling {wires} not supported for gate "
                        f"{name} on {backend.name()}; possible couplings: {couplings}"
                    )


def circuit_to_data(circuit: QuantumCircuit) -> List[List]:
    """Convert the circuit to JSON serializable instructions.

    Helper function that converts a QuantumCircuit into a list of symbolic
    instructions as required by the Json format which is sent to the backend.

    Args:
        circuit: The quantum circuit for which to extract the instructions.

    Returns:
        A list of lists describing the instructions in the circuit. Each sublist
        has three entries the name of the instruction, the wires that the instruction
        applies to and the parameter values of the instruction.
    """

    instructions = []

    for inst in circuit.data:
        name = inst[0].name
        wires = [circuit.qubits.index(qubit) for qubit in inst[1]]
        params = [float(param) for param in inst[0].params]
        instructions.append([name, wires, params])

    return instructions


def circuit_to_cold_atom(
    circuits: Union[List[QuantumCircuit], QuantumCircuit],
    backend: Backend,
    shots: int = 60,
) -> dict:
    """
    Converts a circuit to a JSon payload to be sent to a given backend.

    Args:
        circuits: The circuits that need to be run.
        backend: The backend on which the circuit should be run
        shots: The number of shots for each circuit.

    Returns:
        A list of dicts.
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    # validate the circuits against the backend configuration
    validate_circuits(circuits=circuits, backend=backend, shots=shots)

    experiments = {}
    for idx, circuit in enumerate(circuits):
        experiments["experiment_%i" % idx] = {
            "instructions": circuit_to_data(circuit),
            "shots": shots,
            "num_wires": circuit.num_qubits,
        }

    return experiments
