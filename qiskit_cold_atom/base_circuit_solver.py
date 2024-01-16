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

"""A base class for circuit solvers of cold atomic quantum circuits."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import warnings
import numpy as np
from scipy.sparse import csc_matrix, identity, SparseEfficiencyWarning
from scipy.sparse.linalg import expm

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_cold_atom.exceptions import QiskitColdAtomError


class BaseCircuitSolver(ABC):
    """An abstract class for circuit solvers of different cold atom types.

    By subclassing BaseCircuitSolver one can create circuit solvers for different
    types of cold atomic setups such as spin, fermionic, and bosonic setups. All
    these subclasses will simulate cold atom quantum circuits by exponentiating
    matrices. Therefore, subclasses of BaseCircuitSolver are not intended to solve
    large circuits.
    """

    def __init__(
        self,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        max_dimension: int = 1e6,
        ignore_barriers: bool = True,
    ):
        """
        Args:
            shots: amount of shots for the measurement simulation;
                   if not None, measurements are performed, otherwise no measurements are done.
            seed: seed for the RNG for the measurement simulation
            max_dimension: The maximum Hilbert space dimension (limited to keep
                computation times reasonably short)
            ignore_barriers: If true, will ignore barrier instructions
        """
        self.shots = shots

        self._seed = seed
        if self._seed is not None:
            np.random.seed(self._seed)

        self._max_dimension = max_dimension
        self._ignore_barriers = ignore_barriers
        self._dim = None

    @property
    def seed(self) -> int:
        """The seed for the random number generator of the measurement simulation."""
        return self._seed

    @seed.setter
    def seed(self, value: int):
        """Set the seed for the random number generator. This will also update numpy's seed."""
        np.random.seed(value)
        self._seed = value

    @property
    def max_dimension(self) -> int:
        """The maximal Hilbert space dimension of the simulation."""
        return self._max_dimension

    @max_dimension.setter
    def max_dimension(self, value: int):
        self._max_dimension = value

    @property
    def ignore_barriers(self) -> bool:
        """Boolean flag that defines how barrier instructions in the circuit are handled."""
        return self._ignore_barriers

    @property
    def dim(self) -> int:
        """Return the dimension set by the last quantum circuit on which the solver was called."""
        return self._dim

    @ignore_barriers.setter
    def ignore_barriers(self, boolean):
        self._ignore_barriers = boolean

    def __call__(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Performs the simulation of the circuit: Each operator is converted into a sparse matrix
        over the basis and is then exponentiated to get the unitary of the gate. All these
        unitaries are multiplied to give the total unitary of the circuit. Applying this to the
        initial state yields the final state of the circuit, from which we sample a number `shots`
        of shots (if specified).

        Args:
            circuit: A quantum circuit with gates described by second quantized generators

        Returns:
            output: dict{'unitary' : np.array((dimension, dimension)),
                 'statevector': np.array((dimension, 1)),
                 'counts': dict{string: int}}

        Raises:
            QiskitColdAtomError:
                - If one of the generating Hamiltonians is not hermitian which would
                lead to non-unitary time evolution.
                - If the dimension of the Hilbert space is larger than the max. dimension.
            NotImplementedError:
                - If ignore_barriers is False.
        """

        self.preprocess_circuit(circuit)

        if self._dim > self.max_dimension:
            raise QiskitColdAtomError(
                f"Hilbert space dimension of the simulation ({self._dim}) exceeds the "
                f"maximally supported value {self.max_dimension}."
            )

        # initialize the circuit unitary as an identity matrix
        circuit_unitary = identity(self._dim, dtype=complex)

        for op in self.to_operators(circuit):
            operator_mat = self.operator_to_mat(op)

            # check that the operators are hermitian before exponentiating
            if (operator_mat.H - operator_mat).count_nonzero() != 0:
                raise QiskitColdAtomError("generator of unitary gate is not hermitian!")
            # with the next release of qiskit nature this can be replaced with
            # if not operator.is_hermitian():
            #     raise QiskitColdAtomError("generator of unitary gate is not hermitian!")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
                gate_unitary = expm(-1j * operator_mat)

            circuit_unitary = gate_unitary @ circuit_unitary

        final_state = circuit_unitary @ self.get_initial_state(circuit)

        output = {
            "unitary": circuit_unitary.toarray(),
            "statevector": final_state.toarray().reshape(self._dim),
        }

        # If shots is specified, simulate measurements on the entire register!
        if self.shots is not None:
            meas_distr = np.abs(final_state.toarray().reshape(self._dim)) ** 2

            if not np.isclose(sum(meas_distr), 1.0):
                raise QiskitColdAtomError("Final statevector is not normalized")

            meas_results = self.draw_shots(meas_distr)
            counts_dict = {
                outcome: list(meas_results).count(outcome) for outcome in set(meas_results)
            }
            output["memory"] = meas_results
            output["counts"] = counts_dict

        # return empty memory and counts dictionary if no shots are specified
        else:
            output["memory"] = []
            output["counts"] = {}

        return output

    def to_operators(self, circuit: QuantumCircuit) -> List[SparseLabelOp]:
        """
        Convert a circuit to a list of second quantized operators that describe the generators of the
        gates applied to the circuit. The SparseLabelOps generating the gates are embedded in the
        larger space corresponding to the entire circuit.

        Args:
            circuit: A quantum circuit with gates described by second quantized generators

        Returns:
            operators: a list of second-quantized operators, one for each applied gate, in the order
            of the gates in the circuit

        Raises:
            QiskitColdAtomError: - If a given gate can not be converted into a second-quantized operator
                         - If a gate is applied after a measurement instruction
                         - If a circuit instruction other than a Gate, measure, load or barrier is given
            NotImplementedError: If ignore_barriers is False
        """
        operators = []
        measured = [False] * circuit.num_qubits

        for inst in circuit.data:
            name = inst[0].name
            qargs = [circuit.qubits.index(qubit) for qubit in inst[1]]

            if name == "measure":
                for idx in qargs:
                    measured[idx] = True

            elif name == "load":
                continue

            elif name == "barrier":
                if self.ignore_barriers:
                    continue
                raise NotImplementedError

            elif isinstance(inst[0], Gate):
                try:
                    second_quantized_op = inst[0].generator
                except AttributeError as attribute_error:
                    raise QiskitColdAtomError(
                        f"Gate {inst[0].name} has no defined generator"
                    ) from attribute_error

                if not isinstance(second_quantized_op, SparseLabelOp):
                    raise QiskitColdAtomError(
                        "Gate generator needs to be initialized as qiskit_nature SparseLabelOp"
                    )
                for idx in qargs:
                    if measured[idx]:
                        raise QiskitColdAtomError(
                            f"Simulator cannot handle gate {name} after previous measure instruction."
                        )

                if not second_quantized_op.register_length == len(qargs):
                    raise QiskitColdAtomError(
                        f"length of operator labels {second_quantized_op.register_length} must be "
                        f"equal to length of wires {len(qargs)} the gate acts on"
                    )
                operators.append(
                    self._embed_operator(second_quantized_op, circuit.num_qubits, qargs)
                )

            else:
                raise QiskitColdAtomError(f"Unknown instruction {name} applied to circuit")

        return operators

    @abstractmethod
    def get_initial_state(self, circuit: QuantumCircuit) -> csc_matrix:
        """Returns the initial state of the quantum circuit as a sparse column vector."""

    @abstractmethod
    def _embed_operator(
        self, operator: SparseLabelOp, num_wires: int, qargs: List[int]
    ) -> SparseLabelOp:
        """
        Turning an operator that acts on the wires given in qargs into an operator
        that acts on the entire state space of a circuit. The implementation of the subclasses
        depends on whether the operators use sparse labels (SpinOp) or dense labels (FermionicOp).

        Args:
            operator: SparseLabelOp describing the generating Hamiltonian of a gate
            num_wires: number of wires of the space in which to embed the operator
            qargs: The wire indices the gate acts on

        Returns: A SparseLabelOp acting on the entire quantum register of the Circuit
        """

    @abstractmethod
    def operator_to_mat(self, operator: SparseLabelOp) -> csc_matrix:
        """Turn a SparseLabelOp into a sparse matrix."""

    @abstractmethod
    def preprocess_circuit(self, circuit: QuantumCircuit):
        """Pre-process the circuit, e.g. initialize the basis and validate.
        This needs to update ``dim``, i.e. the Hilbert space dimension of the solver."""

    @abstractmethod
    def draw_shots(self, measurement_distribution: List[float]) -> List[str]:
        """
        Simulates shots by drawing from a given distribution of measurement outcomes.
        Assigning the index of each outcome to the occupations of the modes can be
        non-trivial which is why this step needs to be implemented by the subclasses.

        Args:
            measurement_distribution: A list with the probabilities of the different
                measurement outcomes that has the length of the Hilbert space dimension.

        Returns:
            A list of strings encoding the outcome of the individual shots.
        """
