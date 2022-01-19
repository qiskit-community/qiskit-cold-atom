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

"""Module to convert cold atom circuits to dictionaries that can be sent to backends."""

from typing import List, Union, Optional
from enum import Enum

from qiskit import QuantumCircuit
from qiskit.providers import BackendV1 as Backend
from qiskit_cold_atom.exceptions import QiskitColdAtomError


class CircuitTools:
    """A class to provide tooling for cold-atomic circuits.

    Since all methods are class methods this class does not need to be instantiated. This
    class groups tools for cold atomic circuits. It also makes clear the ordering of the
    fermionic wires that qiskit works with.
    """

    # Qiskit for fermions works with a sequential register definition. For fermionic
    # modes with more than on species the circuits will have a corresponding number of
    # sequential registers with the same length. For example, a three site system with two
    # species will have two sequential registers with three wires each. Other packages may
    # use an "interleaved" wire order.
    __wire_order__ = "sequential"

    @classmethod
    def validate_circuits(
        cls,
        circuits: Union[List[QuantumCircuit], QuantumCircuit],
        backend: Backend,
        shots: Optional[int] = None,
    ) -> None:
        """
        Performs validity checks on circuits against the configuration of the backends. This checks
        whether all applied instructions in the circuit are accepted by the backend and whether the
        applied gates comply with their respective coupling maps.

        Args:
            circuits: The circuits that need to be run.
            backend: The backend on which the circuit should be run.
            shots: The number of shots for each circuit.

        Raises:
            QiskitColdAtomError: If the maximum shot number specified by the backend is exceeded.
            QiskitColdAtomError: If the backend does not support an instruction in the circuit.
            QiskitColdAtomError: If the width of the circuit is too large.
            QiskitColdAtomError: If the circuit has unbound parameters.
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

        config_dict = backend.configuration().to_dict()

        for circuit in circuits:

            try:
                native_gates = {
                    gate.name: gate.coupling_map for gate in backend.configuration().gates
                }
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
            num_species = None
            wire_order = None
            if "num_species" in config_dict:
                num_species = backend.configuration().num_species
                if "wire_order" in config_dict:
                    wire_order = WireOrder(backend.configuration().wire_order)
                else:
                    wire_order = cls.__wire_order__

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

                    if num_species:
                        wires = cls.convert_wire_order(
                            wires,
                            convention_from=cls.__wire_order__,
                            convention_to=wire_order,
                            num_species=num_species,
                            num_sites=circuit.num_qubits//num_species,
                            sort=True,
                        )

                    if wires not in couplings:
                        raise QiskitColdAtomError(
                            f"coupling {wires} not supported for gate "
                            f"{name} on {backend.name()}; possible couplings: {couplings}"
                        )

    @classmethod
    def circuit_to_data(cls, circuit: QuantumCircuit, backend: Backend) -> List[List]:
        """Convert the circuit to JSON serializable instructions.

        Helper function that converts a QuantumCircuit into a list of symbolic
        instructions as required by the Json format which is sent to the backend.

        Args:
            circuit: The quantum circuit for which to extract the instructions.
            backend: The backend on which the circuit should be run.

        Returns:
            A list of lists describing the instructions in the circuit. Each sublist
            has three entries the name of the instruction, the wires that the instruction
            applies to and the parameter values of the instruction.
        """

        instructions = []

        config_dict = backend.configuration().to_dict()
        num_species = None
        wire_order = None
        if "num_species" in config_dict:
            num_species = backend.configuration().num_species
            if "wire_order" in config_dict:
                wire_order = WireOrder(backend.configuration().wire_order)
            else:
                wire_order = cls.__wire_order__

        for inst in circuit.data:
            name = inst[0].name
            wires = [circuit.qubits.index(qubit) for qubit in inst[1]]
            if num_species:
                wires = cls.convert_wire_order(
                    wires,
                    convention_from=cls.__wire_order__,
                    convention_to=wire_order,
                    num_species=num_species,
                    num_sites=circuit.num_qubits//num_species,
                    sort=True,
                )
            params = [float(param) for param in inst[0].params]
            instructions.append([name, wires, params])

        return instructions

    @classmethod
    def circuit_to_cold_atom(
        cls,
        circuits: Union[List[QuantumCircuit], QuantumCircuit],
        backend: Backend,
        shots: int = 60,
    ) -> dict:
        """
        Converts a circuit to a JSon payload to be sent to a given backend.

        Args:
            circuits: The circuits that need to be run.
            backend: The backend on which the circuit should be run.
            shots: The number of shots for each circuit.

        Returns:
            A list of dicts.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        # validate the circuits against the backend configuration
        cls.validate_circuits(circuits=circuits, backend=backend, shots=shots)

        if "wire_order" in backend.configuration().to_dict():
            wire_order = WireOrder(backend.configuration().wire_order)
        else:
            wire_order = cls.__wire_order__

        experiments = {}
        for idx, circuit in enumerate(circuits):
            experiments["experiment_%i" % idx] = {
                "instructions": cls.circuit_to_data(circuit, backend=backend),
                "shots": shots,
                "num_wires": circuit.num_qubits,
                "wire_order": wire_order,
            }

        return experiments

    @classmethod
    def convert_wire_order(
        cls,
        wires: List[int],
        convention_from: WireOrder,
        convention_to: WireOrder,
        num_sites: int,
        num_species: int,
        sort: Optional[bool] = False,
    ) -> List[int]:
        """
        Converts a list of wire indices onto which a gate acts from one convention to another.
        Possible conventions are "sequential", where the first num_sites wires denote the
        first species, the second num_sites wires denote the second species etc., and "interleaved",
        where the first num_species wires denote the first site, the second num_species wires denote
        the second site etc.

        Args:
            wires: Wires onto which a gate acts, e.g. [3, 4, 7, 8].
            convention_from: The convention in which "wires" is given.
            convention_to: The convention into which to convert.
            num_sites: The total number of sites.
            num_species: The number of different atomic species.
            sort: If true, the returned list of indices is sorted in ascending order.

        Raises:
            QiskitColdAtomError: If the convention to and from is not supported.

        Returns:
            A list of wire indices following the convention_to.
        """
        if (convention_to or convention_from) not in WireOrder:
            raise QiskitColdAtomError(
                f"Wire order conversion from {convention_from} to {convention_to}"
                f" is not supported."
            )

        new_wires = None

        if convention_from == convention_to:
            new_wires = wires

        if convention_from == WireOrder.SEQUENTIAL and convention_to == WireOrder.INTERLEAVED:
            new_wires = [idx % num_sites * num_species + idx // num_sites for idx in wires]

        elif convention_from == WireOrder.INTERLEAVED and convention_to == WireOrder.SEQUENTIAL:
            new_wires = [idx % num_species * num_sites + idx // num_species for idx in wires]

        if sort:
            return sorted(new_wires)
        else:
            return new_wires


class WireOrder(str, Enum):
    """The possible wire orderings for cold atomic circuits.

    For example, a sequential register [0, 1, 2, 3, 4, 5] with two species implies that wires 0, 1, 2
    are of the same type while an interleaved ordering implies that wires 0, 2, and 4 are of the
    same type.
    """
    SEQUENTIAL = "sequential"
    INTERLEAVED = "interleaved"
