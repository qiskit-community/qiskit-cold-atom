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

"""General spin simulator backend."""

from typing import Union, List, Dict, Any, Optional
import uuid
from fractions import Fraction
import warnings
import time
import datetime

from qiskit.providers.models import BackendConfiguration
from qiskit.providers import Options
from qiskit.providers.aer import AerJob
from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit.circuit.measure import Measure

from qiskit_cold_atom.spins.spin_circuit_solver import SpinCircuitSolver
from qiskit_cold_atom.spins.base_spin_backend import BaseSpinBackend
from qiskit_cold_atom.circuit_to_cold_atom import validate_circuits


class SpinSimulator(BaseSpinBackend):
    """A simulator to simulate general spin circuits.

    This general spin simulator backend simulates spin circuits with gates that have
    generators described by spin Hamiltonians. It computes the statevector and unitary
    of a circuit and simulates measurements.
    """

    # Default configuration of the backend if the user does not provide one.
    __DEFAULT_CONFIGURATION__ = {
        "backend_name": "spin_simulator",
        "backend_version": "0.0.1",
        "n_qubits": None,
        "basis_gates": None,
        "gates": [],
        "local": True,
        "simulator": True,
        "conditional": False,
        "open_pulse": False,
        "memory": True,
        "max_shots": 1e5,
        "coupling_map": None,
        "description": "a base simulator for spin circuits. Instead of a qubit, each wire represents a "
        "single high-dimensional spin",
    }

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, provider=None):
        """
        Initialize the backend from a configuration dictionary.

        Args:
            config_dict: Configuration dictionary of the backend. If None is given
                a default is assumed.
        """

        if config_dict is None:
            config_dict = self.__DEFAULT_CONFIGURATION__

        super().__init__(
            configuration=BackendConfiguration.from_dict(config_dict), provider=provider
        )

    @classmethod
    def _default_options(cls):
        return Options(shots=1)

    def _execute(self, data: Dict[str, Any], job_id: str = "") -> Result:
        """
        Helper function to execute a job. The circuit and all relevant parameters are
        given in the data dict. Performs validation checks on the received circuits
        and utilizes the FermionCircuitSolver to perform the numerical simulations.

        Args:
            data: Data dictionary that that contains the experiments to simulate, given in the shape:
                data = {
                    "num_species": int,
                    "shots": int,
                    "seed": int,
                    "experiments": Dict[str, QuantumCircuit],
                }
            job_id: The job id assigned by the run method

        Returns:
            result: A qiskit job result.
        """
        # Start timer
        start = time.time()

        output = {"results": []}

        spin = data["spin"]
        shots = data["shots"]
        seed = data["seed"]

        solver = SpinCircuitSolver(spin, shots, seed)

        for exp_i, exp_name in enumerate(data["experiments"]):
            experiment = data["experiments"][exp_name]
            circuit = experiment["circuit"]

            # perform compatibility checks with the backend configuration in case gates and supported
            # instructions are constrained by the backend's configuration
            if self.configuration().gates and self.configuration().supported_instructions:
                validate_circuits(circuits=circuit, backend=self, shots=shots)

            # check whether all wires are measured
            measured_wires = set()

            for inst in circuit.data:
                if isinstance(inst[0], Measure):
                    for wire in inst[1]:
                        index = circuit.qubits.index(wire)
                        if index in measured_wires:
                            warnings.warn(
                                f"Wire {index} has already been measured, "
                                f"second measurement is ignored"
                            )
                        else:
                            measured_wires.add(index)

            if measured_wires and len(measured_wires) != len(circuit.qubits):
                warnings.warn(
                    f"Number of wires in the circuit ({len(circuit.qubits)}) does not equal the "
                    f"number of wires with measurement instructions ({len(measured_wires)}). "
                    f"{self.__class__.__name__} only supports measurement of the entire quantum "
                    "register which will be performed instead."
                )

            if not measured_wires:
                solver.shots = None

            simulation_result = solver(circuit)

            output["results"].append(
                {
                    "header": {"name": exp_name, "random_seed": seed},
                    "shots": shots,
                    "spin": spin,
                    "status": "DONE",
                    "success": True,
                }
            )
            # add the simulation result at the correct place in the result dictionary
            output["results"][exp_i]["data"] = simulation_result

        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name()
        output["backend_version"] = self.configuration().backend_version
        output["time_taken"] = time.time() - start
        output["success"] = True
        output["qobj_id"] = None

        return Result.from_dict(output)

    # pylint: disable=arguments-differ, unused-argument
    def run(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        shots: int = 1000,
        spin: Union[float, Fraction] = Fraction(1, 2),
        seed: Optional[int] = None,
        **run_kwargs,
    ) -> AerJob:
        """
        Run the simulator with a variable length of the individual spins.

        Args:
            circuits: A list of quantum circuits.
            shots: The number of shots to measure.
            spin: The spin length of the simulated system which must be a positive
                integer or half-integer. Defaults to 1/2 which is equivalent to qubits.
            seed: The seed for the simulator.
            run_kwargs: Additional keyword arguments that might be passed down when calling
            qiskit.execute() which will have no effect on this backend.

        Returns:
             aer_job: a job object containing the result of the simulation
        """

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        data = {"spin": spin, "shots": shots, "seed": seed, "experiments": {}}

        for idx, circuit in enumerate(circuits):
            data["experiments"]["experiment_%i" % idx] = {
                "circuit": circuit,
            }

        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._execute, data)
        aer_job.submit()
        return aer_job
