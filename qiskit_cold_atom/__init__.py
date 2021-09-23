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

r"""
.. image:: ../images/qiskit_cold_atom_logo_with_text.svg
   :alt: Missing Logo

==================================================
Qiskit Cold Atom module (:mod:`qiskit_cold_atom`)
==================================================

.. currentmodule:: qiskit_cold_atom

The Qiskit Cold Atom module provides functionality to describe quantum systems of trapped cold atoms
in a gate- and circuit-based framework.

Traditionally, each wire in a quantum circuit represents one qubit as the fundamental unit of information
processing. Here, we extend this concept and allow wires to represent individual internal states of
trapped cold atoms. This currently covers two settings, one for fermionic modes and one for spin
modes.

In a fermionic setting, each wire of a quantum circuit represents an abstract fermionic mode in second
quantization which can either be occupied (1) or empty (0). Such systems are realized experimentally by
individual fermionic atoms trapped in arrays of optical tweezers. Circuit instructions and backends
to interact with and simulate such circuits are given by the :mod:`qiskit_cold_atom.fermions` module.

In a spin setting, each wire of a quantum circuit represents a quantum mechanical spin of a given length
:math:`S`. Upon measurement, each spin is measured in one of its :math:`2S+1` internal basis states
labelled :math:`0` to :math:`2S`, thus it can be thought of as a qudit with dimension :math:`d = 2S+1`.
This setting describes the collective spin of bosonic atoms trapped in a Bose-Einstein-condensate.
Circuit instructions and backends to interact with and simulate such circuits are provided by the
:mod:`qiskit_cold_atom.spins` module.

The quantum circuits that these systems can implement thus utilize a fundamentally different form of
quantum information processing compared to qubits. Therefore, the typical qubit gates can not be applied
to these circuits. Instead, the fermions and spin modules define their own gate sets which are defined
by their second-quantized Hamiltonians that generate the unitary gate. Note that loading the
:mod:`qiskit_cold_atom.fermions` or :mod:`qiskit_cold_atom.spins` module will decorate the
:class:`QuantumCircuit` class in Qiskit by adding methods to call pre-defined fermionic and spin gates,
respectively.

To enable the control of real quantum hardware, the :mod:`qiskit_cold_atom.providers`
module contains a provider which enables access to cold atomic device backends.

The top-level classes and submodules of qiskit_cold_atom are:

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QiskitColdAtomError

Submodules
==========

.. autosummary::
   :toctree:

   applications
   fermions
   providers
   spins

"""
from functools import wraps
from qiskit import QuantumCircuit
from qiskit_cold_atom.exceptions import QiskitColdAtomError


def add_gate(func):
    """Decorator to add a gate method to the QuantumCircuit class"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    setattr(QuantumCircuit, func.__name__, wrapper)

    return func


__version__ = "0.1.0"

__all__ = ["__version__", "QiskitColdAtomError", "add_gate"]
