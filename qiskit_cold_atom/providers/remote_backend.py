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

"""Classes for remote cold atom backends."""

import json
from typing import List, Dict, Union
import requests

from qiskit.providers.models import BackendConfiguration
from qiskit.providers import Options
from qiskit import QuantumCircuit
from qiskit.providers import ProviderV1 as Provider
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import JobStatus

from qiskit_cold_atom.spins.base_spin_backend import BaseSpinBackend
from qiskit_cold_atom.fermions.base_fermion_backend import BaseFermionBackend
from qiskit_cold_atom.circuit_tools import CircuitTools
from qiskit_cold_atom.providers.cold_atom_job import ColdAtomJob
from qiskit_cold_atom.exceptions import QiskitColdAtomError


class RemoteBackend(Backend):
    """Remote cold atom backend."""

    def __init__(self, provider: Provider, url: str):
        """
        Initialize the backend by querying the server for the backend configuration dictionary.

        Args:
            provider: The provider which need to have the correct credentials attributes in place
            url: The url of the backend server

        Raises:
            QiskitColdAtomError: If the connection to the backend server can not be established.

        """

        self.url = url
        self.username = provider.credentials["username"]
        self.token = provider.credentials["token"]
        # Get the config file from the remote server
        try:
            r = requests.get(
                self.url + "/get_config",
                params={
                    "username": self.username,
                    "password": self.token,
                },
            )
        except requests.exceptions.ConnectionError as err:
            raise QiskitColdAtomError(
                "connection to the backend server can not be established."
            ) from err

        super().__init__(configuration=BackendConfiguration.from_dict(r.json()), provider=provider)

    @classmethod
    def _default_options(cls) -> Options:
        """Return the default options.

        Returns:
            qiskit.providers.Options: A options object with default values set
        """
        return Options(shots=1)

    @property
    def credentials(self) -> Dict[str, Union[str, List[str]]]:
        """Returns: the access credentials used."""
        return self.provider().credentials

    # pylint: disable=arguments-differ, unused-argument
    def run(
        self,
        circuit: Union[QuantumCircuit, List[QuantumCircuit]],
        shots: int = 1,
        **run_kwargs,
    ) -> ColdAtomJob:
        """
        Run a quantum circuit or list of quantum circuits.

        Args:
            circuit: The quantum circuits to be executed on the device backend
            shots: The number of measurement shots to be measured for each given circuit
            run_kwargs: Additional keyword arguments that might be passed down when calling
            qiskit.execute() which will have no effect on this backend.

        Raises:
            QiskitColdAtomError: If the response from the backend does not have a job_id.

        Returns:
            A Job object through the backend can be queried for status, result etc.
        """

        job_payload = CircuitTools.circuit_to_cold_atom(circuit, self, shots=shots)

        res = requests.post(
            self.url + "/post_job/",
            data={
                "json": json.dumps(job_payload),
                "username": self.username,
                "password": self.token,
            },
        )

        res.raise_for_status()
        response = res.json()

        if "job_id" not in response:
            raise QiskitColdAtomError("The response has no job_id.")

        return ColdAtomJob(self, response["job_id"])

    def retrieve_job(self, job_id: str) -> ColdAtomJob:
        """Return a single job submitted to this backend.

        Args:
            job_id: The ID of the job to retrieve.

        Returns:
            The job with the given ID.

        Raises:
            QiskitColdAtomError: If the job retrieval failed.
        """

        retrieved_job = ColdAtomJob(backend=self, job_id=job_id)

        try:
            job_status = retrieved_job.status()
        except requests.exceptions.RequestException as request_error:
            raise QiskitColdAtomError(
                "connection to the remote backend could not be established"
            ) from request_error

        if job_status == JobStatus.ERROR:
            raise QiskitColdAtomError(f"Job with id {job_id} could not be retrieved")

        return retrieved_job


class RemoteSpinBackend(RemoteBackend, BaseSpinBackend):
    """Remote backend which runs spin circuits."""


class RemoteFermionBackend(RemoteBackend, BaseFermionBackend):
    """Remote backend which runs fermionic circuits."""
