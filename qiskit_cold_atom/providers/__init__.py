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

"""
Cold Atom Backend Provider
===========================

This module contains a provider to interface with cold atomic device backends.
To access these devices, the user has to specify the url of the backend, a username and an access token.
With these credentials, the provider can be accessed like this:

.. code-block:: python

    from qiskit_cold_atom.providers import ColdAtomProvider

    # save account with personal credentials
    ColdAtomProvider.save_account(url="my_url", username="my_name", token="my_url")

    provider = ColdAtomProvider.load_account()
    backends = provider.backends()

.. autosummary::
   :toctree: ../stubs/

    ColdAtomProvider


Cold Atom Jobs
==============

When executing jobs, cold atom backends will return instances of :class:`ColdAtomJob`.

.. autosummary::
   :toctree: ../stubs/

    ColdAtomJob

"""

from .cold_atom_provider import ColdAtomProvider
from .cold_atom_job import ColdAtomJob
