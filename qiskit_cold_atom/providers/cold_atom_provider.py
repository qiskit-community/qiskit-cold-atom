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

"""Provider for cold atomic experiments."""

import os
import warnings
import json
from configparser import ConfigParser, ParsingError
from typing import Callable, Dict, Optional, Union, List
import errno

import requests

from qiskit.providers import ProviderV1 as Provider
from qiskit.providers import BackendV1 as Backend
from qiskit.providers.providerutils import filter_backends
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit_cold_atom.exceptions import QiskitColdAtomError

from qiskit_cold_atom.fermions import FermionSimulator
from qiskit_cold_atom.spins import SpinSimulator

from qiskit_cold_atom.providers.fermionic_tweezer_backend import (
    FermionicTweezerSimulator,
)
from qiskit_cold_atom.providers.remote_backend import (
    RemoteSpinBackend,
    RemoteFermionBackend,
)
from qiskit_cold_atom.providers.collective_spin_backend import CollectiveSpinSimulator

# Default location of the credentials file
_DEFAULT_COLDATOM_FILE = os.path.join(
    os.path.expanduser("~"), ".qiskit", "cold_atom_credentials.conf"
)


class ColdAtomProvider(Provider):
    """
    Provider for cold atomic backends. To access the devices, the user has to specify the url of the
    backend, a username and an access token.
    With these credentials, the provider can be accessed like this:

    .. code-block:: python

        from qiskit_cold_atom.providers import ColdAtomProvider
        # save account with personal credentials
        ColdAtomProvider.save_account(url="my_url", username="my_name", token="my_url")
        # load account
        provider = ColdAtomProvider.load_account()
        # access backends
        backends = provider.backends()
    """

    def __init__(self, credentials: Optional[Dict[str, Union[str, List[str]]]] = None):
        """
        Constructor of the cold atom provider. Using the given credentials, the provider tries to
        access the remote backends and adds all backends that can be reached to the available backends.

        Args:
            credentials: A dictionary of the shape
            {'urls' = ['http://...', ...] 'username' = '...', 'token' = '...'}
            The value to the url key could also be a single url

        Raises:
            QiskitColdAtomError: If the credentials file specified does not match the expected format
        """

        self.credentials = credentials
        self.name = "qiskit_cold_atom_provider"

        # Populate the list of backends with the local simulators
        backends = [
            FermionSimulator(provider=self),
            SpinSimulator(provider=self),
            FermionicTweezerSimulator(provider=self),
            CollectiveSpinSimulator(provider=self),
        ]

        if credentials is not None:

            try:
                urls = self.credentials["urls"]
                name = self.credentials["username"]
                token = self.credentials["token"]
            except KeyError as key_err:
                raise QiskitColdAtomError(
                    "Credentials do not match the expected format"
                ) from key_err
            if isinstance(urls, str):
                urls = [urls]

            # try to ping remote backend with credentials:
            for url in urls:
                try:
                    r = requests.get(
                        url + "/get_config",
                        params={"username": name, "token": token},
                    )

                    backend_type = r.json()["cold_atom_type"]

                    if backend_type == "fermion":
                        backend = RemoteFermionBackend
                    elif backend_type == "spin":
                        backend = RemoteSpinBackend
                    else:
                        warnings.warn(
                            f"specified cold atom type {backend_type} at url {url} is not recognized"
                        )
                        continue

                    backends.append(backend(provider=self, url=url))

                except requests.exceptions.RequestException as request_error:
                    warnings.warn(
                        f"connection to the remote backend could not be established: {request_error}"
                    )
                except json.JSONDecodeError as json_error:
                    warnings.warn(
                        f"Backend configuration could not be read: {r.text}, {json_error}"
                    )
                except KeyError:
                    warnings.warn(f"backend at url {url} has no specified cold atom type")

        self._backends = BackendService(backends)

    def __str__(self):
        return f"<{self.__class__.__name__}(name={self.name})>"

    def __repr__(self):
        return self.__str__()

    def get_backend(self, name: str = None, **kwargs) -> Backend:
        """Return a single backend matching the specified filtering.
        Args:
            name: name of the backend.
            **kwargs: dict used for filtering.
        Returns:
            backend: a backend matching the filtering.
        Raises:
            QiskitBackendNotFoundError: if no backend could be found or
                more than one backend matches the filtering criteria.
        """
        backends = self._backends(name, **kwargs)
        if len(backends) > 1:
            raise QiskitBackendNotFoundError("More than one backend matches criteria.")
        if not backends:
            raise QiskitBackendNotFoundError("No backend matches criteria.")

        return backends[0]

    @staticmethod
    def save_account(
        url: Union[str, List[str]],
        username: str,
        token: str,
        overwrite: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """Save credentials to a file locally

        Args:
            url: The url(s) of the backends(s) to connect to
            username: username of the account
            token: the token of the account
            overwrite: If true, will overwrite any credentials already stored on disk
            filename: Full path to the credentials file. If ``None``, the default
                location is used (``$HOME/.qiskit/cold_atom_credentials``).

        Raises:
            OSError: If there is a race condition when creating the directory for the
                credentials if it does not already exist.
        """
        if isinstance(url, str):
            url = [url]

        filename = filename or _DEFAULT_COLDATOM_FILE

        credentials_present = False

        if os.path.isfile(filename):

            stored_credentials = ColdAtomProvider.stored_account(filename=filename)

            if stored_credentials != {}:
                credentials_present = True

            if credentials_present and not overwrite:
                warnings.warn("Credentials already present. Set overwrite=True to overwrite.")
        else:
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        if not credentials_present or overwrite:

            credentials = {
                "cold-atom-credentials": {
                    "urls": " ".join(url),
                    "username": username,
                    "token": token,
                }
            }
            # Write the credentials file.
            with open(filename, "w") as credentials_file:
                config_parser = ConfigParser()
                config_parser.read_dict(credentials)
                config_parser.write(credentials_file)

    @staticmethod
    def add_url(
        url: str,
        filename: Optional[str] = None,
    ) -> None:
        """Add a url to the existing credentials stored on disk

        Args:
            url: The url of a backend to add to the existing credentials
            filename: Full path to the credentials file. If ``None``, the default
                location is used (``$HOME/.qiskit/cold_atom_credentials``).

        Raises:
            QiskitColdAtomError: If there are no stored credentials at the specified location
        """
        credentials_dict = ColdAtomProvider.stored_account(filename)
        try:
            urls = credentials_dict["urls"]
            name = credentials_dict["username"]
            token = credentials_dict["token"]
        except KeyError as err:
            raise QiskitColdAtomError("No stored credentials found") from err
        urls.append(url)
        ColdAtomProvider.save_account(urls, name, token, overwrite=True, filename=filename)

    @classmethod
    def load_account(cls, filename: Optional[str] = None) -> "ColdAtomProvider":
        """Check for stored credentials and return a provider instance. If no credentials are found
        the provider will not include the remote backends bu only the simulators

        Args:
            filename: Full path to the credentials file. If ``None``, the default
                location is used (``$HOME/.qiskit/cold_atom_credentials``).

        Returns:
            A provider instance initialized with the backends available to the account
        """

        stored_credentials = ColdAtomProvider.stored_account(filename=filename)
        if stored_credentials == {}:
            warnings.warn("No stored credentials found")
            return cls()
        else:
            return cls(stored_credentials)

    @classmethod
    def enable_account(
        cls, url: Union[str, List[str]], username: str, token: str
    ) -> "ColdAtomProvider":
        """Create provider from credentials given at runtime without storing credentials

        Args:
            url: The url(s) of the backends(s) to connect to
            username: username of the account
            token: the token of the account

        Returns:
            A provider instance initialized with the backends available to the account
        """
        if isinstance(url, str):
            url = [url]
        credentials = {"urls": url, "username": username, "token": token}
        return cls(credentials)

    @staticmethod
    def delete_account(filename: Optional[str] = None) -> None:
        """Delete any credentials saved locally at filename. This will not delete the file itself but
        erase the content of that file

        Args:
            filename: Full path to the credentials file. If ``None``, the default
                location is used (``$HOME/.qiskit/cold_atom_credentials``).
        """
        filename = filename or _DEFAULT_COLDATOM_FILE

        if not os.path.isfile(filename):
            warnings.warn("No credentials found at the specified location")

        else:
            # Write an empty dictionary to the file
            with open(filename, "w") as credentials_file:
                config_parser = ConfigParser()
                config_parser.read_dict({})
                config_parser.write(credentials_file)

    @staticmethod
    def stored_account(
        filename: Optional[str] = None,
    ) -> Dict[str, Union[str, List[str]]]:
        """Retrieve the credentials stored on disk.

        Args:
            filename: Full path to the credentials file. If ``None``, the default
                location is used (``$HOME/.qiskit/cold_atom_credentials``).

        Raises:
            QiskitColdAtomError: If the file specified does not match the expected format

        Returns:
            A dictionary containing the found credentials. This dictionary is empty if no credentials
            are found at the specified locations
        """
        filename = filename or _DEFAULT_COLDATOM_FILE
        config_parser = ConfigParser()
        try:
            config_parser.read(filename)
        except ParsingError as ex:
            raise QiskitColdAtomError(f"Error parsing file {filename}: {str(ex)}") from ex

        credentials_dict = {}

        for name in config_parser.sections():
            if name.startswith("cold-atom-credentials"):
                credentials_dict = dict(config_parser.items(name))
                credentials_dict["urls"] = credentials_dict["urls"].split()
            else:
                raise QiskitColdAtomError(f"stored credentials file has unknown section {name}")

        return credentials_dict

    def active_account(self) -> Optional[Dict[str, str]]:
        """Return the credentials in use for the session."""
        return self.credentials

    # pylint: disable=arguments-differ
    def backends(self, name: Optional[str] = None, filters: Optional[Callable] = None, **kwargs):
        """Abstract method of the Base Provider class"""
        return self._backends(name, filters, **kwargs)


class BackendService:
    """A service class that allows for autocompletion
    of backends from provider.
    """

    def __init__(self, backends):
        """Initialize service
        Parameters:
            backends (list): List of backend instances.
        """
        self._backends = backends
        for backend in backends:
            setattr(self, backend.name(), backend)

    def __call__(self, name: str = None, filters: Callable = None, **kwargs):
        """A listing of all backends from this provider.
        Parameters:
            name: The name of a given backend.
            filters: A filter function.
        Returns:
            list: A list of backends, if any.
        """
        # pylint: disable=arguments-differ
        backends = self._backends
        if name:
            backends = [backend for backend in backends if backend.name() == name]

        return filter_backends(backends, filters=filters, **kwargs)
