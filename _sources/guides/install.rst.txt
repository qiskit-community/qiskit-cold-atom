Installation
============

You can install qiskit cold atoms using pip:

.. code-block:: python

    pip install qiskit-cold-atoms

Provider Setup
--------------

You can instantiate the provider by running

.. code-block:: python

    provider = ColdAtomProvider.load_account()

which will load any cold-atom saved credentials that you have.
If no credentials have been stored this will provide the
:class:`FermionicTweezerSimulator` and :class:`CollectiveSpinSimulator`
simulators.
If you have a username, URL, and token for a cold-atom setup you can save
your credentials by running

.. code-block:: python

    ColdAtomProvider.save_account(urls=list_of_urls, username="JohnDoe",token="123456")
