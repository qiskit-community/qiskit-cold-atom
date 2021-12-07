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
Module to support spin circuits.

The spins module holds the circuit instructions and simulators needed for cold-atom setups which
control the collective spin of an ensemble of atoms in a Bose-Einstein-condensate.

In this setting, each wire in a quantum circuit describes a single quantum mechanical angular momentum
(or spin) with principal quantum number :math:`S`. Upon measurement, each wire can be found in a state
ranging from :math:`0` to :math:`2S`.
Backends to describe such spin circuits are subclasses of the :class:`BaseSpinBackend` class.
The :class:`SpinSimulator` backend is a general purpose simulator backend that simulates spin
circuits.

Spin backends
-------------------
.. autosummary::
   :toctree: ../stubs/

   BaseSpinBackend
   SpinSimulator

At the start of the circuit each spin is taken to be initialized in the 0 state, in analogy to qubits.


Spin gates
==========

Spin gates are quantum circuit instructions designed specifically for cold-atom based setups that
control large spin ensembles. These gates are characterized by their effect on the spin ensemble.
Spin gates are subclasses or instances of the :class:`SpinGate` class. All of these gates define a
``generator`` property used to compute the time-evolution. These generators are second quantized
operators (:class:`SpinOp` from Qiskit Nature) that describe the gate Hamiltonian acting on the spins.
When an entry of the :mod:`qiskit_cold_atom.spins` module is imported, the Spin gates are added to the
:class:`QuantumCircuit` class in Qiskit.

The module includes a number of gates suitable to a platform that implements rotations and squeezing of
collective spins.

.. autosummary::
   :toctree: ../stubs/

    SpinGate
    RLXGate
    RLYGate
    RLZGate
    RLZ2Gate
    OATGate
    RLZLZGate
    RLXLYGate

These gates should serve as an example of how a concrete collective spin platform can be described
through :mod:`qiskit_cold_atom`.
Users are encouraged to define their own gates to describe different collective spin experiments.
If these gates define a ``generator`` property as laid out above, the :class:SpinSimulator
can be used to simulate circuits with such custom gates.


Circuit solvers
===============
Circuit solvers are classes that allow users to simulate a quantum circuit
for cold-atom based setups. They are subclasses of
:class:`BaseCircuitSolver` and can be called on quantum circuits to solve them.
The numerical simulation of spin circuits is carried out by the :class:`SpinCircuitSolver` class.
This simulates the circuits via exact diagonalization and provides access to the unitary,
the statevector and simulated measurement outcomes of the circuit.

.. autosummary::
   :toctree: ../stubs/

   SpinCircuitSolver
"""

from .spin_simulator_backend import SpinSimulator
from .base_spin_backend import BaseSpinBackend
from .spin_circuit_solver import SpinCircuitSolver

# Gate imports
from .spins_gate_library import (
    SpinGate,
    RLXGate,
    RLYGate,
    RLZGate,
    RLZ2Gate,
    OATGate,
    RLZLZGate,
    RLXLYGate,
)
