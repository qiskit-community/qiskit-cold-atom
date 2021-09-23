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

"""
Module to study applications using cold atomic setups. Users of the qiskit_cold_atoms.applications
module can create Fermionic time-evolution problems to simulate their dynamics on simulator
backends or cold-atom based hardware that natively supports the Hamiltonian of the fermionic
problem.

For lattice-based problems, users must create subclasses of the :class:`FermionicLattice` class.
An example of which is the one-dimensional :class:`FermiHubbard1D` which describes a
one-dimensional Fermi-Hubbard lattice with spin-up and spin-down particles.

.. autosummary::
   :toctree: ../stubs/

    FermionicLattice
    FermiHubbard1D
    FermionicEvolutionProblem
    TimeEvolutionSolver

"""

from .fermi_hubbard import FermionicLattice, FermiHubbard1D
from .fermionic_evolution_problem import FermionicEvolutionProblem
from .time_evolution_solver import TimeEvolutionSolver
