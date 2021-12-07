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

"""General Fermion simulator backend."""

from typing import Union, List, Dict, Any, Optional
import uuid
import warnings
import time
import datetime

from qiskit.providers.models import BackendConfiguration
from qiskit.providers import Options
from qiskit.providers.aer import AerJob
from qiskit import QuantumCircuit
from qiskit.result import Result

from qiskit_cold_atom.fermions.fermion_circuit_solver import FermionCircuitSolver
from qiskit_cold_atom.fermions.base_fermion_backend import BaseFermionBackend
from qiskit_cold_atom.circuit_to_cold_atom import validate_circuits


class FermionSimulator(BaseFermionBackend):
    """A simulator to simulate general fermionic circuits.

    This general fermion simulator backend simulates fermionic circuits with gates that have
    generators described by fermionic Hamiltonians. It computes the statevector and unitary
    of a circuit and simulates measurements.
    """

    _DEFAULT_CONFIGURATION = {
        "backend_name": "fermion_simulator",
        "backend_version": "0.0.1",
        "n_qubits": 20,
        "basis_gates": None,
        "gates": [],
        "local": False,
        "simulator": True,
        "conditional": False,
        "open_pulse": False,
        "memory": True,
        "max_shots": 1e5,
        "coupling_map": None,
        "description": "a base simulator for fermionic circuits. Instead of qubits, each wire represents"
        " a single fermionic mode",
        "supported_instructions": None,
    }

    def __init__(self, config_dict: Dict[str, Any] = None, provider=None):
        """Initializing the backend from a configuration dictionary"""

        if config_dict is None:
            config_dict = self._DEFAULT_CONFIGURATION

        super().__init__(
            configuration=BackendConfiguration.from_dict(config_dict), provider=provider
        )

    @classmethod
    def _default_options(cls):
        return Options(shots=1)

    def _execute(self, data: Dict[str, Any], job_id: str = ""):
        """Helper function to execute a job. The circuit and all relevant parameters are given in the
        data dict. Performs validation checks on the received circuits and utilizes
        the FermionCircuitSolver to perform the numerical simulations.

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

        num_species = data["num_species"]
        shots = data["shots"]
        seed = data["seed"]

        solver = FermionCircuitSolver(num_species=num_species, shots=shots, seed=seed)

        for exp_i, exp_name in enumerate(data["experiments"]):
            experiment = data["experiments"][exp_name]
            circuit = experiment["circuit"]

            # perform compatibility checks with the backend configuration in case gates and supported
            # instructions are constrained by the backend's configuration
            if self.configuration().gates and self.configuration().supported_instructions:
                validate_circuits(circuits=circuit, backend=self, shots=shots)

            # check whether all wires are measured
            measured_wires = []

            for inst in circuit.data:
                name = inst[0].name

                if name == "measure":

                    for wire in inst[1]:
                        index = circuit.qubits.index(wire)
                        if index in measured_wires:
                            warnings.warn(
                                f"Wire {index} has already been measured, "
                                f"second measurement is ignored"
                            )
                        else:
                            measured_wires.append(index)

            if measured_wires and len(measured_wires) != len(circuit.qubits):
                warnings.warn(
                    f"Number of wires in circuit ({len(circuit.qubits)}) exceeds number of wires "
                    + f" with assigned measurement instructions ({len(measured_wires)}). "
                    + "This simulator backend only supports measurement of the entire quantum register "
                    "which will instead be performed."
                )

            # If there are no measurements, set shots to None
            if not measured_wires:
                solver.shots = None

            simulation_result = solver(circuit)

            output["results"].append(
                {
                    "header": {"name": exp_name, "random_seed": seed},
                    "shots": shots,
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
        seed: Optional[int] = None,
        num_species: int = 1,
        **run_kwargs,
    ) -> AerJob:
        """
        Method to run circuits on the backend.

        Args:
            circuits: QuantumCircuit applying fermionic gates to run on the backend
            shots: Number of measurement shots taken in case the circuit has measure instructions
            seed: seed for the random number generator of the measurement simulation
            num_species: number of different fermionic species described by the circuits
            run_kwargs: Additional keyword arguments that might be passed down when calling
                qiskit.execute() which will have no effect on this backend.

        Returns:
            aer_job: a job object containing the result of the simulation
        """

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        data = {
            "num_species": num_species,
            "shots": shots,
            "seed": seed,
            "experiments": {},
        }

        for idx, circuit in enumerate(circuits):
            data["experiments"][f"experiment_{idx}"] = {
                "circuit": circuit,
            }

        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._execute, data)
        aer_job.submit()
        return aer_job

    @staticmethod
    def get_basis(circuit: QuantumCircuit, num_species: int = 1):
        """Get the basis of fermionic states in occupation number representation for the simulation
        of a given quantum circuit.

        Args:
            circuit: A quantum circuit using Fermionic Gates.
            num_species: Number of different fermionic species described by the circuit.

        Returns:
            basis: the fermionic basis in which the simulation of the circuit is performed.
        """
        solver = FermionCircuitSolver(num_species=num_species)
        solver.preprocess_circuit(circuit)
        basis = solver.basis

        return basis

    def draw(self, qc: QuantumCircuit, **draw_options):
        """Draw the circuit by defaulting to the draw method of QuantumCircuit.

        Note that in the future this method may be modified and tailored to fermion
        quantum circuits.

        Args:
            qc: The quantum circuit to draw.
            draw_options: Key word arguments for the drawing of circuits.
        """
        qc.draw(**draw_options)
