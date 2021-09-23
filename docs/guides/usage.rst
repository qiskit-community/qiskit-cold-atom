===================
Submitting circuits
===================

Quantum circuits for cold atomic backends should be built from the library of gates included in
the fermions and spin packages. These circuits can be sent to a backend. These gates are
automatically added to Qiskit's QuantumCircuit class when importing the cold atom provider.
The code below shows an example of a simulation of a rotation of a system with a size 20 spin.

.. code-block:: python

    import numpy as np

    from qiskit.circuit import QuantumCircuit
    from qiskit.visualization import plot_histogram
    from qiskit_cold_atom.providers import ColdAtomProvider

    provider = ColdAtomProvider()
    backend = provider.get_backend("collective_spin_simulator")

    circuit = QuantumCircuit(1, 1)
    circuit.lx(np.pi/2, 0)
    circuit.measure(0, 0)

    job_rabi = backend.run(circuit, shots=1024, spin=20, seed=5462)
    plot_histogram(job_rabi.result().get_counts(0))


Basis gates and transpilation
=============================

The cold atom backends supported in this package are typically not universal quantum computers
but quantum simulators with gates that implement the hardware-native Hamiltonian.
Therefor, the circuits to submit to the hardware must be built with the fermion and spin gates
provided by the package.

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_cold_atom.spins.spins_gate_library import LXGate, LZGate, LZ2Gate

    circ = QuantumCircuit(1, 1)
    circ.lx(-np.pi/2, 0)
    circ.lz2(0.3, 0)
    circ.lz(-np.pi/2, 0)
    circ.lx(-0.15, 0)
    circ.measure(0, 0)

Additional details are in the tutorials.
