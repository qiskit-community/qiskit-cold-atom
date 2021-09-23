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
Module to support fermionic circuits.

The fermions module holds the circuit instructions and simulators needed
by cold-atom setups that trap fermionic atoms in optical tweezer arrays or lattices.

In this setting, each wire in a quantum circuit describes a fermionic mode. Upon measurement, each mode
can be found to be occupied by a single particle (1) or be empty (0).
Backends to describe such fermionic circuits are subclasses of the :class:`BaseFermionBackend` class.
The :class:`FermionSimulator` backend is a general purpose simulator backend that simulates fermionic
circuits similar to the QasmSimulator for qubits.

Fermionic backends
-------------------
.. autosummary::
   :toctree: ../stubs/

   BaseFermionBackend
   FermionSimulator

The fermions might also come in several distinguishable species, as is the case when they carry a spin
degree of freedom. In this case, each spatial mode of an experiment can be occupied by a particle
of each spin state. In the circuit description, each individual mode is assigned its own wire.
For example, a system of spin-1/2 fermions in four spatial modes is described by a circuit with eight
wires where the first four wires denote the spin-up and the last four wires denote the spin-down modes.

Prior to applying gates, the fermionic modes in a quantum circuit need to be initialized with particles,
which defines the total number of particles (excitations) in the circuit. This initial occupation number
state can then be manipulated by applying Fermionic gates.

Fermionic gates
----------------
Fermionic gates are quantum circuit instructions designed specifically for
cold-atom based setups that control fermionic atoms in tweezers. These gates are
characterized by their effect on the fermions. Fermionic gates are subclasses or instances of the
:class:`FermionicGate` class. All of these gates define a ``generator`` property used to compute
the time-evolution. These generators are second quantized operators (:class:`FermionicOp` from Qiskit
Nature) that describe the gate Hamiltonian acting on the register.
When an entry of the :mod:`qiskit_cold_atom.fermions` module is imported, the Fermionic Gates are added
to the :class:`QuantumCircuit` class in Qiskit.

The module includes a number of gates suitable to a platform that natively implements Fermi-Hubbard type
dynamics.

.. autosummary::
   :toctree: ../stubs/

    FermionicGate
    LoadFermions
    LocalPhase
    Hopping
    Interaction
    FermiHubbard
    FermionRX
    FermionRY
    FermionRZ

These gates should serve as an example of how a concrete fermionic platform can be described
through :mod:`qiskit_cold_atom`.
Users are encouraged to define their own gates to describe different fermionic hardware.
If these gates define a ``generator`` property as laid out above, the :class:`FermionSimulator`
can be used to simulate circuits with such custom gates.

Circuit Solver
----------------
Circuit solvers are classes that allow users to simulate a quantum circuit
for cold-atom based setups. They are subclasses of
:class:`BaseCircuitSolver` and can be called on quantum circuits to solve them.
The numerical simulation of fermionic circuits is carried out by the :class:`FermionCircuitSolver` class.
This simulates the circuits via exact diagonalization and provides access to the unitary,
the statevector and simulated measurement outcomes of the circuit.

.. autosummary::
   :toctree: ../stubs/

   FermionCircuitSolver

"""

from qiskit_cold_atom.fermions.fermion_simulator_backend import FermionSimulator
from qiskit_cold_atom.fermions.base_fermion_backend import BaseFermionBackend
from qiskit_cold_atom.fermions.fermion_circuit_solver import FermionCircuitSolver

from qiskit_cold_atom.fermions.fermion_gate_library import (
    FermionicGate,
    LoadFermions,
    LocalPhase,
    Hopping,
    Interaction,
    FermiHubbard,
    FermionRX,
    FermionRY,
    FermionRZ,
)
