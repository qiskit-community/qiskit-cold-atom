from qiskit.circuit.gate import Gate
from qiskit_cold_atom import QiskitColdAtomError, add_gate
import numpy as np

class LoadGate(Gate):
    """The load gate."""

    def __init__(self,num_atoms:int) -> None:
        """Create a new gate.

        Args:
            params: A list of parameters.
        """

        super().__init__(name="load", num_qubits=1, params=[num_atoms], label=None)

@add_gate
def load(self, wire, num_atoms):
    # pylint: disable=invalid-name
    """add the load gate to a QuantumCircuit"""
    return self.append(LoadGate(num_atoms), [wire], [])

class RLXGate(Gate):
    """The rLx gate."""

    def __init__(self,omega:float) -> None:
        """Create a new gate.

        Args:
            params: A list of parameters.
        """

        super().__init__(name="rLx", num_qubits=1, params=[omega], label=None)

@add_gate
def rLx(self, wire, omega):
    # pylint: disable=invalid-name
    """add the rLx gate to a QuantumCircuit"""
    return self.append(RLXGate(omega), [wire], [])

class RLZGate(Gate):
    """The rLz gate."""

    def __init__(self,delta:float) -> None:
        """Create a new gate.

        Args:
            params: A list of parameters.
        """

        super().__init__(name="rLz", num_qubits=1, params=[delta], label=None)

@add_gate
def rLz(self, wire, delta):
    # pylint: disable=invalid-name
    """add the rLz gate to a QuantumCircuit"""
    return self.append(RLZGate(delta), [wire], [])

class RLZ2Gate(Gate):
    """The rLz2 gate."""

    def __init__(self,chi:float) -> None:
        """Create a new gate.

        Args:
            params: A list of parameters.
        """

        super().__init__(name="rLz2", num_qubits=1, params=[chi], label=None)

@add_gate
def rLz2(self, wire, chi):
    # pylint: disable=invalid-name
    """add the rLz2 gate to a QuantumCircuit"""
    return self.append(RLZ2Gate(chi), [wire], [])