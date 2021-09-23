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

"""A solver for time-evolution problems."""

from typing import List

from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.mappers.second_quantization import (
    JordanWignerMapper,
    BravyiKitaevMapper,
    ParityMapper,
)
from qiskit_nature.converters.second_quantization import QubitConverter

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.opflow.evolutions import PauliTrotterEvolution
from qiskit import execute

from qiskit_cold_atom.applications.fermionic_evolution_problem import (
    FermionicEvolutionProblem,
)
from qiskit_cold_atom.fermions.base_fermion_backend import BaseFermionBackend


class TimeEvolutionSolver:
    """
    Solver class that solves time evolution problem by either analog simulation on fermionic
    hardware or trotterized time evolution on qubit hardware. The computation that this time
    evolution solver will do depends on the type of the backend.
    """

    MAPPER_DICT = {
        "bravyi_kitaev": BravyiKitaevMapper(),
        "jordan_wigner": JordanWignerMapper(),
        "parity": ParityMapper(),
    }

    def __init__(
        self,
        backend,
        map_type: str = None,
        trotter_steps: int = None,
        shots: int = 1000,
    ):
        """
        Initialize a time evolution solver

        Args:
            backend: The backend on which to execute the problem, may be qubit or fermionic.
            map_type: The fermion-to-qubit mapping required if a qubit backend is given
            trotter_steps: The amount of trotter steps to approximate time evolution on
                qubit backends
            shots: number of measurements taken of the constructed circuits
        """

        self.backend = backend
        self.map_type = map_type
        self.trotter_steps = trotter_steps
        self.shots = shots

    def solve(self, problem: FermionicEvolutionProblem) -> List[float]:
        """Solve the problem using the provided backend

        Args:
            problem: The FermionicEvolutionProblem to solve.

        Returns:
            A list of expectation values of the observable of the problem. This list has the
            same length as the list of times for which to compute the time evolution.
        """

        if isinstance(self.backend, BaseFermionBackend):
            qc_load = self.backend.initialize_circuit(problem.initial_state.occupations)
            circuits = problem.circuits(qc_load)

            observable_evs = self.backend.measure_observable_expectation(
                circuits, problem.observable, self.shots
            )

        else:
            # use qubit pipeline
            circuits = self.construct_qubit_circuits(problem)

            # construct observable
            mapper = self.MAPPER_DICT[self.map_type]
            qubit_observable = mapper.map(problem.observable)
            observable_mat = qubit_observable.to_spmatrix()

            observable_evs = [0.0] * len(problem.evolution_times)

            for idx, circuit in enumerate(circuits):

                circuit.measure_all()

                job = execute(circuit, self.backend, shots=self.shots)
                counts = job.result().get_counts().int_outcomes()

                for outcome_ind in counts:
                    prob = counts[outcome_ind] / self.shots

                    observable_evs[idx] += (
                        prob * observable_mat.diagonal()[outcome_ind].real
                    )

        return observable_evs

    def construct_qubit_circuits(
        self, problem: FermionicEvolutionProblem
    ) -> List[QuantumCircuit]:
        """Convert the problem to a trotterized qubit circuit using the specified map_type

        Args:
            problem: The fermionic evolution problem specifying the system, evolution-time
                and observable to be measured

        Returns:
            a list of quantum circuits that simulate the time evolution.
            There is one circuit for each evolution time specified in the problem'
        """

        psi_0 = problem.initial_state
        system = problem.system
        hamiltonian = system.to_fermionic_op()

        mapper = self.MAPPER_DICT[self.map_type]

        circuits = []

        # construct circuit of initial state:
        label = ["+" if bit else "I" for bit in psi_0.occupations_flat]
        bitstr_op = FermionicOp("".join(label))
        qubit_op = QubitConverter(mapper).convert(bitstr_op)[0]
        init_circ = QuantumCircuit(QuantumRegister(qubit_op.num_qubits, "q"))
        # Add gates in the right positions: we are only interested in the `X` gates because we want
        # to create particles (0 -> 1) where the initial state introduced a creation (`+`) operator.
        for i, bit in enumerate(qubit_op.primitive.table.X[0]):
            if bit:
                init_circ.x(i)

        for time in problem.evolution_times:

            # time-step of zero will cause PauliTrotterEvolution to fail
            if time == 0.0:
                time += 1e-10

            # map fermionic hamiltonian to qubits
            qubit_hamiltonian = mapper.map(hamiltonian * time)
            # get time evolution operator by exponentiating
            exp_op = qubit_hamiltonian.exp_i()
            # perform trotterization

            evolved_op = PauliTrotterEvolution(reps=self.trotter_steps).convert(exp_op)

            trotter_circ = evolved_op.to_circuit_op().to_circuit()

            circuits.append(init_circ.compose(trotter_circ))

        return circuits
