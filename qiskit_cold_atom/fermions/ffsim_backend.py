# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fermionic simulator backend that uses ffsim."""

import datetime
import time
import uuid
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import ffsim
import numpy as np
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.providers import Options
from qiskit.providers.models import BackendConfiguration
from qiskit.result import Result
from qiskit_aer import AerJob
from qiskit_nature.operators.second_quantization import FermionicOp
from scipy.sparse.linalg import expm_multiply

from qiskit_cold_atom.circuit_tools import CircuitTools
from qiskit_cold_atom.fermions.base_fermion_backend import BaseFermionBackend
from qiskit_cold_atom.fermions.fermion_gate_library import (
    FermionicGate,
    Hop,
    Interaction,
    LoadFermions,
    Phase,
)


class FfsimBackend(BaseFermionBackend):
    """Fermionic simulator backend that uses ffsim."""

    _DEFAULT_CONFIGURATION = {
        "backend_name": "ffsim_simulator",
        "backend_version": "0.0.1",
        "n_qubits": 100,
        "basis_gates": None,
        "gates": [],
        "local": False,
        "simulator": True,
        "conditional": False,
        "open_pulse": False,
        "memory": True,
        "max_shots": 1e6,
        "coupling_map": None,
        "description": "ffsim simulator for fermionic circuits. Instead of qubits, each wire represents"
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
        ffsim to perform the numerical simulations.

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
        assert num_species == 2
        shots = data["shots"]
        seed = data["seed"]

        for exp_i, exp_name in enumerate(data["experiments"]):
            experiment = data["experiments"][exp_name]
            circuit = experiment["circuit"]

            # perform compatibility checks with the backend configuration in case gates and supported
            # instructions are constrained by the backend's configuration
            if self.configuration().gates and self.configuration().supported_instructions:
                CircuitTools.validate_circuits(circuits=circuit, backend=self, shots=shots)

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
                shots = None

            simulation_result = _simulate_ffsim(circuit, shots, seed)

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
        num_species: int = 2,
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
        assert num_species == 2

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


def _simulate_ffsim(circuit: QuantumCircuit, shots: int | None = None, seed=None) -> dict[str, Any]:
    assert circuit.num_qubits % 2 == 0
    norb = circuit.num_qubits // 2
    occ_a, occ_b = _get_initial_occupations(circuit)
    nelec = len(occ_a), len(occ_b)
    vec = ffsim.slater_determinant(norb, (occ_a, occ_b))
    qubit_indices = {q: i for i, q in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        op, qubits, _ = instruction.operation, instruction.qubits, instruction.clbits
        if isinstance(op, Hop):
            orbs = [qubit_indices[q] for q in qubits]
            spatial_orbs = _get_spatial_orbitals(orbs, norb)
            vec = _simulate_hop(
                vec, np.array(op.params), spatial_orbs, norb=norb, nelec=nelec, copy=False
            )
        elif isinstance(op, Interaction):
            orbs = [qubit_indices[q] for q in qubits]
            spatial_orbs = _get_spatial_orbitals(orbs, norb)
            (interaction,) = op.params
            vec = _simulate_interaction(
                vec, interaction, spatial_orbs, norb=norb, nelec=nelec, copy=False
            )
        elif isinstance(op, Phase):
            orbs = [qubit_indices[q] for q in qubits]
            spatial_orbs = _get_spatial_orbitals(orbs, norb)
            vec = _simulate_phase(
                vec, np.array(op.params), spatial_orbs, norb=norb, nelec=nelec, copy=False
            )
        elif isinstance(op, FermionicGate):
            orbs = [qubit_indices[q] for q in qubits]
            spatial_orbs = _get_spatial_orbitals(orbs, norb)
            ferm_op = _fermionic_op_to_fermion_operator(op.generator, spatial_orbs)
            linop = ffsim.linear_operator(ferm_op, norb, nelec)
            # TODO use ferm_op.values once it's available
            scale = sum(abs(ferm_op[k]) for k in ferm_op)
            vec = expm_multiply(-1j * linop, vec, traceA=scale)
            # TODO remove this
            np.testing.assert_allclose(np.linalg.norm(vec), 1.0)

    result = {"statevector": vec}

    if shots is None:
        result["memory"] = []
        result["counts"] = {}
    else:
        rng = np.random.default_rng(seed)
        probs = np.abs(vec) ** 2
        samples = rng.choice(np.arange(len(vec)), size=shots, replace=True, p=probs)
        bitstrings = ffsim.indices_to_strings(samples, norb, nelec)
        result["memory"] = bitstrings
        result["counts"] = Counter(bitstrings)

    return result


def _get_initial_occupations(circuit: QuantumCircuit):
    norb = circuit.num_qubits // 2
    occ_a, occ_b = set(), set()
    occupations = [occ_a, occ_b]
    active_qubits = set()
    for instruction in circuit.data:
        if isinstance(instruction.operation, LoadFermions):
            for q in instruction.qubits:
                if q in active_qubits:
                    raise ValueError(
                        f"Encountered Load instruction on qubit {q} after it has "
                        "already been operated on."
                    )
                spin, orb = divmod(circuit.qubits.index(q), norb)
                # reverse index due to qiskit convention
                occupations[spin].add(norb - 1 - orb)
        else:
            active_qubits |= set(instruction.qubits)
    return tuple(occ_a), tuple(occ_b)


def _get_spatial_orbitals(orbs: list[int], norb: int) -> list[int]:
    assert len(orbs) % 2 == 0
    alpha_orbs = orbs[: len(orbs) // 2]
    beta_orbs = [orb - norb for orb in orbs[len(orbs) // 2 :]]
    assert alpha_orbs == beta_orbs
    # reverse orbitals due to qiskit convention
    alpha_orbs = [norb - 1 - orb for orb in alpha_orbs]
    return alpha_orbs


def _simulate_hop(
    vec: np.ndarray,
    coeffs: np.ndarray,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    copy: bool,
) -> np.ndarray:
    mat = np.zeros((norb, norb))
    for i, val in zip(range(len(target_orbs) - 1), coeffs):
        j, k = target_orbs[i], target_orbs[i + 1]
        if j < k:
            val = -val
        mat[j, k] = -val
        mat[k, j] = -val
    coeffs, orbital_rotation = scipy.linalg.eigh(mat)
    return ffsim.apply_num_op_sum_evolution(
        vec, coeffs, 1.0, norb=norb, nelec=nelec, orbital_rotation=orbital_rotation, copy=copy
    )


def _simulate_interaction(
    vec: np.ndarray,
    interaction: float,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    copy: bool,
) -> np.ndarray:
    mat_alpha_beta = np.zeros((norb, norb))
    mat_alpha_beta[target_orbs, target_orbs] = interaction
    return ffsim.apply_diag_coulomb_evolution(
        vec,
        mat=np.zeros((norb, norb)),
        mat_alpha_beta=mat_alpha_beta,
        time=1.0,
        norb=norb,
        nelec=nelec,
        copy=copy,
    )


def _simulate_phase(
    vec: np.ndarray,
    mu: np.ndarray,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    copy: bool,
) -> np.ndarray:
    coeffs = np.zeros(norb)
    coeffs[target_orbs] = mu
    return ffsim.apply_num_op_sum_evolution(
        vec, coeffs, time=1.0, norb=norb, nelec=nelec, copy=copy
    )


def _fermionic_op_to_fermion_operator(
    op: FermionicOp, target_orbs: list[int]
) -> ffsim.FermionOperator:
    """Convert a Qiskit Nature FermionicOp to an ffsim FermionOperator."""
    norb_small = len(target_orbs)
    coeffs = {}
    for term, coeff in op.terms():
        fermion_actions = []
        for action_str, index in term:
            action = action_str == "+"
            spin, orb = divmod(index, norb_small)
            fermion_actions.append((action, bool(spin), target_orbs[orb]))
        coeffs[tuple(fermion_actions)] = coeff
    return ffsim.FermionOperator(coeffs)
