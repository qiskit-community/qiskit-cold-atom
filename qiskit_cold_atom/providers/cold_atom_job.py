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

"""Job for cold atom backends."""

from typing import Dict, Optional
import time
import json
import requests

from qiskit.providers import BackendV1 as Backend
from qiskit.providers import JobV1 as Job
from qiskit.providers import JobTimeoutError, JobError
from qiskit.providers import JobStatus
from qiskit.result import Result


class ColdAtomJob(Job):
    """Class of jobs returned by cold atom backends."""

    def __init__(self, backend: Backend, job_id: str):
        """
        Args:
            backend: The backend on which the job was run.
            job_id: The ID of the job.
        """
        super().__init__(backend, job_id)

        self.token = self._backend.token
        self.user = self._backend.username

    def _wait_for_result(self, timeout: float = None, wait: float = 5.0) -> Dict:
        """
        Query the backend to get the result.

        Args:
            timeout: time after which the server is no longer queried for the result
            wait: waiting time between queries for the result of the backend

        Returns:
            result dictionary formatted according to Qiskit schemas.

        Raises:
            JobTimeoutError: If the timeout is reached without receiving the result
            JobError: If the returned status reports an error
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if timeout and elapsed >= timeout:
                raise JobTimeoutError("Timed out waiting for result")

            result = requests.get(
                self._backend.url + "/get_job_result",
                params={
                    "job_id": self._job_id,
                    "username": self.user,
                    "password": self.token,
                },
            ).json()

            if result["status"] == "finished":
                break
            if result["status"] == "error":
                raise JobError(f"{self._backend.name} returned error:\n" + str(result))
            time.sleep(wait)

        return result

    # pylint: disable=arguments-differ
    def result(self, timeout: Optional[float] = None, wait: float = 5.0) -> Result:
        """
        Retrieve a qiskit result object from the backend.

        Args:
            timeout: time after which the server is no longer queried for the result
            wait: waiting time between queries for the result of the backend

        Returns:
            qiskit.result object that contains the outcome of the job
        """
        result_dict = self._wait_for_result(timeout, wait=wait)

        return Result.from_dict(result_dict)

    def status(self):
        """
        Retrieve the status from the backend.

        Returns:
            status: A string describing the status of the job.
        """

        r = requests.get(
            self._backend.url + "/get_job_status",
            params={
                "job_id": self.job_id(),
                "username": self.user,
                "password": self.token,
            },
        )

        status_string = r.json()["status"]

        # If the backend can not be reached return ERROR as a status
        if r.status_code != 200:
            status = JobStatus.ERROR
        elif status_string == "INITIALIZING":
            status = JobStatus.INITIALIZING
        elif status_string == "QUEUED":
            status = JobStatus.QUEUED
        elif status_string == "VALIDATING":
            status = JobStatus.VALIDATING
        elif status_string == "RUNNING":
            status = JobStatus.RUNNING
        elif status_string == "CANCELLED":
            status = JobStatus.CANCELLED
        elif status_string == "DONE":
            status = JobStatus.DONE
        else:
            status = JobStatus.ERROR

        return status

    def error_message(self) -> Optional[str]:
        """
        Retrieve the error message from the backend.

        Returns:
            error: A string describing the error that happened on the backend.
        """
        status_payload = {"job_id": self.job_id()}

        r = requests.get(
            self._backend.url + "/get_job_status/",
            params={
                "json": json.dumps(status_payload),
                "username": self.user,
                "password": self.token,
            },
        )

        return r.json().get("error_message", None)

    def cancel(self):
        raise NotImplementedError

    def submit(self):
        raise NotImplementedError
