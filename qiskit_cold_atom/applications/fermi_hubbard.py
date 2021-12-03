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

"""Module to build a Fermi-Hubbard problem."""

from abc import ABC, abstractmethod
from typing import List

from qiskit import QuantumCircuit
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_cold_atom.fermions.fermion_circuit_solver import FermionicBasis
from qiskit_cold_atom.fermions.fermion_gate_library import FermiHubbard
from qiskit_cold_atom.exceptions import QiskitColdAtomError


class FermionicLattice(ABC):
    """Abstract base fermionic lattice."""

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of lattice sites of the system."""

    @abstractmethod
    def to_fermionic_op(self) -> FermionicOp:
        """Creates the Hamiltonian of the lattice in second quantization.

        Returns:
            The Hamiltonian as a FermionicOp.
        """

    @abstractmethod
    def to_circuit(self, time: float = 1.0) -> QuantumCircuit:
        """
        Wrap the generator of the system in a QuantumCircuit.

        Args:
            time: Duration of the time evolution.

        Returns:
            A quantum circuit which corresponds to the time-evolved Hamiltonian.
        """


class FermiHubbard1D(FermionicLattice):
    """Describes a one-dimensional Fermi-Hubbard model with open boundary conditions."""

    def __init__(
        self,
        num_sites: int,
        particles_up: int,
        particles_down: int,
        hop_strength: float,
        int_strength: float,
        potential: List[float],
    ):
        r"""
        Initialize a one-dimensional fermi-hubbard system. In second quantization this system is
        described by the Hamiltonian

        :math:`H = \sum_{i=1,\sigma}^{L-1} - J_i (f^\dagger_{i,\sigma} f_{i+1,\sigma} +
        f^\dagger_{i+1,\sigma} f_{i,\sigma}) + U \sum_{i=1}^{L} n_{i,\uparrow} n_{i,\downarrow}
        +  \sum_{i=1,\sigma}^{L} \mu_i n_{i,\sigma}`

        Args:
            num_sites: number of lattice sites in the 1D chain.
            particles_up: total number of spin-up particles in the lattice
            particles_down: total number of spin-down particles in the lattice
            hop_strength: strength of hopping between sites
            int_strength: strength of the local interaction
            potential: list of local phases, must be on length num_wires

        Raises:
            QiskitColdAtomError: if the length of the potential does not match the system size.
        """

        # pylint: disable=invalid-name
        self._size = num_sites
        self.particles_up = particles_up
        self.particles_down = particles_down
        self.J = hop_strength
        self.U = int_strength
        self.basis = FermionicBasis(self.size, n_particles=[self.particles_up, self.particles_down])

        if not len(potential) == self.size:
            raise QiskitColdAtomError(
                f"The length of the potentials {len(potential)} must match system size {self.size}"
            )

        self.mu = potential

    @property
    def size(self) -> int:
        """Return the number of sites of the problem."""
        return self._size

    def to_fermionic_op(self) -> FermionicOp:
        """Construct the hamiltonian of the lattice as a FermionicOp.

        Returns:
            A FermionicOp defining the systems Hamiltonian
        """

        operator_labels = []

        # add hopping terms
        for idx in range(self.size - 1):

            right_to_left_up = "I" * idx + "+-" + "I" * (self.size * 2 - idx - 2)
            operator_labels.append((right_to_left_up, -self.J))
            left_to_right_up = "I" * idx + "-+" + "I" * (self.size * 2 - idx - 2)
            operator_labels.append((left_to_right_up, self.J))
            right_to_left_down = "I" * (self.size + idx) + "+-" + "I" * (self.size - idx - 2)
            operator_labels.append((right_to_left_down, -self.J))
            left_to_right_down = "I" * (self.size + idx) + "-+" + "I" * (self.size - idx - 2)
            operator_labels.append((left_to_right_down, self.J))

        # add interaction terms
        for idx in range(self.size):
            opstring = "I" * idx + "N" + "I" * (self.size - 1) + "N" + "I" * (self.size - 1 - idx)
            operator_labels.append((opstring, self.U))

        # add potential terms
        for idx in range(self.size):
            op_up = "I" * idx + "N" + "I" * (2 * self.size - idx - 1)
            operator_labels.append((op_up, self.mu[idx]))
            op_down = "I" * (self.size + idx) + "N" + "I" * (self.size - idx - 1)
            operator_labels.append((op_down, self.mu[idx]))

        return FermionicOp(operator_labels)

    def to_circuit(self, time: float = 1.0) -> QuantumCircuit:
        """
        Wrap the generator of the system in a QuantumCircuit.

        Args:
            time: Duration of the time evolution.

        Returns:
            A quantum circuit which corresponds to the time-evolved Hamiltonian.
        """
        circ = QuantumCircuit(2 * self.size)
        circ.append(
            FermiHubbard(
                num_modes=2 * self.size,
                j=[self.J * time] * (self.size - 1),
                u=self.U * time,
                mu=[mu_i * time for mu_i in self.mu],
            ),
            qargs=range(2 * self.size),
        )

        return circ
