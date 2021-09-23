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

"""Module to describe a basis of fermionic states in occupation number representation."""

from typing import List, Union
from itertools import combinations, chain, product
import numpy as np

from qiskit_nature.operators.second_quantization import FermionicOp

from qiskit_cold_atom.fermions.fermionic_state import FermionicState


class FermionicBasis:
    """Class that represents the basis states of the fermionic fock space in occupation number
    representation for given particle numbers. The ordering of states complies with Qiskit bitstring
    ordering where the smaller number in binary representation has a lower index in the basis"""

    def __init__(
        self,
        sites: int,
        n_particles: Union[int, List[int]],
        particle_conservation: bool = True,
        spin_conservation: bool = True,
    ):
        """
        Args:
            sites: number of spatial fermionic modes
            n_particles: the total number of particles. If given as a list, the entries of the list
                give the particles per spin species, where the length of the list defines the number
                of different fermionic species
            particle_conservation: Boolean flag for the conservation of the total particle number
            spin_conservation: Boolean flag for conservation of the particle number per spin species
        """

        self.sites = sites

        if isinstance(n_particles, int):
            n_particles = [n_particles]

        self.n_particles = n_particles
        self.n_tot = sum(n_particles)
        self.num_species = len(n_particles)

        states = []

        if particle_conservation:
            if spin_conservation:

                indices = []
                for i, n in enumerate(self.n_particles):
                    indices.append(list(combinations(np.arange(sites) + i * sites, n)))

                for combination in product(*indices):
                    particle_indices = list(_ for _ in chain.from_iterable(combination))

                    occupations = [0] * self.sites * self.num_species
                    for idx in particle_indices:
                        occupations[idx] = 1

                    states.append(
                        FermionicState.from_total_occupations(
                            occupations, self.num_species
                        )
                    )

            else:
                for indices_tot in list(
                    combinations(range(self.num_species * sites), self.n_tot)
                ):
                    occupations_tot = [0] * sites * self.num_species
                    for i in indices_tot:
                        occupations_tot[i] = 1
                    states.append(
                        FermionicState.from_total_occupations(
                            occupations_tot, self.num_species
                        )
                    )
        else:
            for occs in product("10", repeat=(self.num_species * self.sites)):
                occupations_tot = [int(n) for n in occs]
                states.append(FermionicState(list(occupations_tot)))

        # reverse order of states to comply with Qiskit bitstring ordering
        self.states = states[::-1]

        self.dimension = len(self.states)

    def __str__(self):
        string = ""
        if self.dimension < 30:
            for i in range(self.dimension):
                if i < 10:
                    string += "\n {}.   ".format(i) + self.states[i].__str__()
                else:
                    string += "\n {}.  ".format(i) + self.states[i].__str__()
        else:
            for i in range(5):
                string += "\n {}.  ".format(i) + self.states[i].__str__()
            string += "\n . \n . \n ."
            for i in range(self.dimension - 5, self.dimension):
                string += "\n {}.  ".format(i) + self.states[i].__str__()

        return string

    @classmethod
    def from_state(
        cls,
        state: FermionicState,
        spin_conservation: bool,
        particle_conservation: bool = True,
    ):
        """Helper function to create the basis corresponding to a given occupation number state with
        particle number conservation and optionally spin conservation."""
        sites = state.sites
        n_particles = []
        for occs in state.occupations:
            n_particles.append(sum(occs))

        return cls(sites, n_particles, particle_conservation, spin_conservation)

    @classmethod
    def from_fermionic_op(cls, fer_op: FermionicOp):
        """Helper function to create the full Fock space basis corresponding to a given FermionicOp."""
        sites = fer_op.register_length
        n_particles = fer_op.register_length
        return cls(
            sites, n_particles, particle_conservation=False, spin_conservation=False
        )

    def get_occupations(self) -> List[List[int]]:
        """Get a list of the flattened occupations of the individual basis states."""
        return [state.occupations_flat for state in self.states]

    def get_index_of_measurement(self, bitstring: str) -> int:
        """
        For a binary string of occupations, e.g. '10100011', return the index of the basis state
        that corresponds to these occupations. In contrast to qubits, this index is not given by 2
        to the power of the bitstring as fermionic bases do not always include all states due to
        particle and spin conservation rules.

        Args:
            bitstring: A binary string of occupations.

        Returns:
            The index of the basis state corresponding to the bitstring.
        """
        occupation_strings = ["".join(map(str, k)) for k in self.get_occupations()]
        index = occupation_strings.index(bitstring)
        return index
