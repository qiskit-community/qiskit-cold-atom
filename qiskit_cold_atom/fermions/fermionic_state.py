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

"""Module to describe fermionic states in occupation number basis"""

from typing import List, Union
import warnings
import numpy as np

from qiskit import QuantumCircuit

from qiskit_cold_atom.exceptions import QiskitColdAtomError


class FermionicState:
    """Fermionic states in an occupation number representation."""

    def __init__(self, occupations: Union[List[int], List[List[int]]]):
        """Create a :class:`FermionicState` from the given occupations.

        Args:
            occupations: List of occupation numbers. When List[int] is given, the occupations
            correspond to the number of indistinguishable fermionic particles in each mode,
            e.g. [0, 1, 1, 0] implies that sites one and two are occupied by a fermion.
            When List[List[int]] is given, the occupations describe the number of particles in
            fermionic modes with different (distinguishable) species of fermions. Each
            inner list gives the occupations of one fermionic species.

        Raises:
            QiskitColdAtomError:
                - If the inner lists do not have the same length
                - If the occupations are not 0 or 1
        """

        if isinstance(occupations[0], int):
            occupations = [occupations]

        self._sites = len(occupations[0])
        self._occupations = occupations
        self._num_species = len(occupations)

        self._occupations_flat = []
        for occs in self.occupations:
            self._occupations_flat += occs

        for occs in self.occupations[0:]:
            if len(occs) != self._sites:
                raise QiskitColdAtomError(
                    f"All occupations of different fermionic species must have "
                    f"same length, received {self.occupations[0]} and {occs}."
                )
            for n in occs:
                if n not in (0, 1):
                    raise QiskitColdAtomError(f"Fermionic occupations must be 0 or 1, got {n}.")

    @property
    def occupations(self) -> List[List[int]]:
        """Return the occupation number of each fermionic mode."""
        return self._occupations

    @property
    def occupations_flat(self) -> List[int]:
        """Return the occupations of each fermionic mode in a flat list."""
        return self._occupations_flat

    @property
    def sites(self) -> int:
        """Return the number of fermionic sites."""
        return self._sites

    @property
    def num_species(self) -> int:
        """Return the number of species of fermions, e.g. 2 for spin up/down systems."""
        return self._num_species

    def __str__(self):
        output = ""
        for i in range(self.num_species):
            output += "|" + str(self.occupations[i])[1:-1] + ">"
        return output

    @classmethod
    def from_total_occupations(cls, occupations: List[int], num_species: int) -> "FermionicState":
        """
        Create a fermionic state from a single (flat) list of total occupations.

        Args:
            occupations: a list of occupations of all fermionic modes, e.g. [0, 1, 1, 0, 1, 0].
            num_species: number of fermionic species. If > 1, the total occupation list is cast
                into a nested list where each inner list describes one fermionic species. In the
                above example, for num_species = 2, this becomes
                FermionicState([[0, 1, 1], [0, 1, 0]]).

        Returns:
            A fermionic state initialized with the given input.

        Raises:
            QiskitColdAtomError: If the length of occupations is not a multiple of num_species.
        """
        if len(occupations) % num_species != 0:
            raise QiskitColdAtomError(
                "The state must have a number of occupations that is a multiple of the"
                "number of fermionic species."
            )

        sites = int(len(occupations) / num_species)
        return cls(np.reshape(occupations, (num_species, sites)).tolist())

    @classmethod
    def initial_state(cls, circuit: QuantumCircuit, num_species: int = 1) -> "FermionicState":
        """
        Create a fermionic state from a quantum circuit that uses the `LoadFermion` instruction.
        This instruction must be the first instructions of the circuit and no further LoadFermion
        instruction can be applied, even after other instructions such as gates have been applied.

        Args:
            circuit: a quantum circuit with LoadFermions instructions that initialize fermionic
                particles.
            num_species: number of different fermionic species, e.g. 1 for a single
                type of spinless fermions (default), 2 for spin-1/2 fermions etc.

        Returns:
            A FermionicState initialized from the given circuit.

        Raises:
            QiskitColdAtomError:
                - If the number of wires in the circuit is not a multiple of num_species,
                - If LoadFermions instructions come after other instructions.
        """
        if num_species > 1:
            if circuit.num_qubits % num_species != 0:
                raise QiskitColdAtomError(
                    "The circuit must have a number of wires that is a multiple of the"
                    "number of fermionic species."
                )

        occupations = [0] * circuit.num_qubits
        gates_applied = [False] * circuit.num_qubits

        if not circuit.data[0][0].name == "load":
            warnings.warn(
                "No particles have been initialized, the circuit will return a trivial result."
            )

        # check that there are no more 'LoadFermions' instructions
        for instruction in circuit.data:
            qargs = [circuit.qubits.index(qubit) for qubit in instruction[1]]

            if instruction[0].name == "load":
                for idx in qargs:
                    if gates_applied[idx]:
                        raise QiskitColdAtomError(
                            f"State preparation instruction in circuit after gates on wire {idx}"
                        )
                    occupations[idx] = 1
            else:
                for idx in qargs:
                    gates_applied[idx] = True

        return cls.from_total_occupations(occupations, num_species)
