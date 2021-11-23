from typing import Union, Optional, List

from qiskit.circuit.gate import Gate
from qiskit_cold_atom import QiskitColdAtomError, add_gate
import numpy as np

class LoadGate(Gate):
    """The load gate."""

    def __init__(self) -> None:
        """Create a new gate.

        Args:
            params: A list of parameters.
        """

        super().__init__(name="load", num_qubits=1, params=[], label=None)

@add_gate
def load(self, wire):
    # pylint: disable=invalid-name
    """add the load gate to a QuantumCircuit"""
    return self.append(LoadGate(), [wire], [])

class InterGate(Gate):
    """The inter gate."""

    def __init__(self,wires:List[int], theta_u:float) -> None:
        """Create a new gate.

        Args:
            params: A list of parameters.
        """
        assert len(wires) == 8, 'InterGate must act on all modes which is 8.'

        super().__init__(name="int", num_qubits=len(wires), params=[theta_u], label=None)

@add_gate
def inter(self, wires:List[int], theta_u:float):
    # pylint: disable=invalid-name
    """add the inter gate to a QuantumCircuit"""
    return self.append(InterGate(wires,theta_u), wires)

class HopGate(Gate):
    """The hop gate."""

    def __init__(self,wires:List[int], theta_j:float) -> None:
        """Create a new gate.

        Args:
            params: A list of parameters.
        """
        assert wires[0]%2 == 0, 'HopGate modes start with an even index.'
        assert wires == list(range(wires[0],wires[0]+4)), 'HopGate must act on nearest neighbor tweezers following interleaved notation.'

        super().__init__(name="hop", num_qubits=len(wires), params=[theta_j], label=None)

@add_gate
def hop(self, wires:List[int], theta_j:float):
    # pylint: disable=invalid-name
    """add the hop gate to a QuantumCircuit"""
    return self.append(HopGate(wires,theta_j), wires)

class PhaseGate(Gate):
    """The phase gate."""

    def __init__(self,wires:List[int], theta_mu:float) -> None:
        """Create a new gate.

        Args:
            params: A list of parameters.
        """
        assert wires[0]%2 == 0, 'PhaseGate modes start with an even index.'
        assert wires == list(range(wires[0],wires[0]+2)), 'PhaseGate must act on one tweezer following interleaved notation.'

        super().__init__(name="phase", num_qubits=len(wires), params=[theta_mu], label=None)

@add_gate
def phase(self, wires:List[int], theta_mu:float):
    # pylint: disable=invalid-name
    """add the phase gate to a QuantumCircuit"""
    return self.append(PhaseGate(wires,theta_mu), wires)