#############################
Backend communication schemas
#############################

``qiskit-cold-atom`` and the backends communicate by exchanging data over a REST API.
This tutorial outlines the schemas that are sent to, and received from the backend. 
They have the following general steps, which we describe in more detail below.


1. Obtain the backend configuration through through a ``GET`` request at the endpoint ``get_config``.
2. Post the job to the backend through a ``POST`` request at the endpoint ``post_job`` .
3. Verify the job status through a ``GET`` request at the endpoint ``get_job_status``.
4. Obtain the result  through a ``GET`` request at the endpoint ``get_job_result``.

We will now discuss each step in more detail.

Backend configuration
~~~~~~~~~~~~~~~~~~~~~

``qiskit-cold-atom`` must have some information from the backend to be able to prepare the
quantum circuits that will be run on the backend.
Each backend has a unique URL assigned to it from which the configuration is retrieved.
The cold atom backend is initialized by making a ``GET`` request to the endpoint ``get_config``, using the python ``requests`` package,
to this URL which retrieves the configuration data of the backend provided as a Json file.

The configuration is created by calling ``BackendConfiguration.from_dict(r.json())`` where
``r`` is the result from the ``GET`` request.
The experimental setup must therefore return a JSon file that complies with the Qiskit schemas
found in ``backend_configuration_schema.json``.
The configuration schemas comprise all relevant information needed for working with the backend.
As an instructive example for the structure of this Json file, we show below what a configuration
could look like for the atomic mixture experimental backend.

.. parsed-literal::

    {
      'backend_name': 'atomic_mixtures',
      'backend_version': '0.0.1',
      'n_qubits': 2,     # number of wires
      'atomic_species': ['Na', 'Li'],
      'basis_gates': ['delay', 'rx'],
      'gates': [
        {
          'name': 'delay',
          'parameters': ['tau', 'delta'],
          'qasm_def': 'gate delay(tau, delta) {}',
          'coupling_map': [[0, 1]],
          'description': 'evolution under SCC Hamiltonian for time tau'
        },
        {
          'name': 'rx',
          'parameters': ['theta'],
          'qasm_def': 'gate rx(theta) {}',
          'coupling_map': [[0]],
          'description': 'Rotation of the sodium spin'
        }
      ],
      'supported_instructions': ['delay', 'rx', 'measure', 'barrier'],
      'local': False,            # backend is local or remote (as seen from user)
      'simulator': False,        # backend is a simulator
      'conditional': False,      # backend supports conditional operations
      'open_pulse': False,       # backend supports open pulse
      'memory': True,            # backend supports memory
      'max_shots': 60,
      'coupling_map': [[0, 1]],
      'max_experiments': 3,
      'description': 'Setup of an atomic mixtures experiment with one trapping site and two atomic species, namely Na and Li.',
      'url': 'http://url_of_the_remote_server',
      'credits_required': False,
      'online_date': datetime.pyi,
      'display_name': str = None
    }

We now discuss some of the entries in this JSon file in more detail.


- ``n_qubits`` gives the maximum amount of wires (which are Qubit objects in Qiskit, hence the name)
  that an incoming circuit can have. In this case, we describe the experiment with one wire per
  atomic species, so ``n_qubits = 2``.
  In the case of ``qiskit-cold-atom`` the entry ``n_qubits`` is a misnomer but we use it here to comply
  to the Qiskit schemas.
  The fact that it is a misnomer does not impact ``qiskit-cold-atom``.

- ``basis_gates`` is a list of the supported instructions. These are defined via the ``GateConfig``
  class whose sepcifications are given as a dictionary under the key ``gates``.
  Note that in the ``qasm_def``, the actual specification ``{}`` remains empty.
  The ``coupling_map`` defines on which wires (``Qubit`` s) these gates can be executed.
  As the ``rx`` gate can only be applied to the sodium wire, its ``coupling_map`` is ``[[0]]``.

- ``supported_instructions`` is a list of the names of all instructions that can be applied to a
  circuit to be run on this backend. This includes the names of all the gates but also non-unitary
  instructions such as measurements and barriers.

- For now we set ``open_pulse`` to ``False``.
  This means that the backend will not accept ``PulseJob`` s which are defined in ``OpenPulse``.
  This functionality may be added later to give users pulse-level access to the hardware.

- The maximum amount of shots for a single quantum circuit in one job is given by ``max_shots``.
  A job can consist of multiple quantum circuits that a user wants to execute.
  The maximum number of different circuits per job is thus bounded by ``max_experiments``.

- The ``url`` of the backend gives the network address of the remote server to which the user
  communicates their job requests.

- Some backends may wish to manage how many jobs a user can run. This can be done by requesting
  that users have sufficient credits to run jobs. If credits are required then the
  ``credits_required`` keyword is set to ``True``.

Job payload
~~~~~~~~~~~

This section describes the information that is sent to the backends to run quantum circuits.
If you want to expose a cold-atomic setup as a Qiskit backend then it must be capable of accepting
these payloads.
The ``run`` method of a backend converts the circuits into a Json serializable ``dict`` and sends it
to the backend url via a ``POST`` request to the ``post_job`` endpoint.
Additional parameters are sent along in the ``json`` dict of the request.
The circuits are converted into the ``dict`` by the ``circuit_to_cold_atom()`` function.
This ``dict`` has the structure shown below.
Each circuit has a unique identifier ``experiment_id``.
Each circuit is then specified by its ``data`` where ``inst_name`` is a reserved name and takes on
the value of one of the basis instructions supported by the hardware as communicated by its configuration file.
``wires`` indicates the wires in the circuit that the instruction is applied to.
``wires`` therefore indicates the trapping site and atomic species to which the instruction is applied.
Finally, ``params`` is the list of parameter values, e.g. a rotation angle, that the instruction takes.
``shots`` defines the number of times the circuit is repeated.

.. parsed-literal::
    {
      experiment_id(str): {
        'instructions': [
          (inst_name(str), wires(List[int]), params(List[float])),
        ],
        'shots': int,
        'num_wires': int
      }
    }

As example consider the circuit data below which could be received by the NaLi device backend as a Json file.
The instructions in data show that this circuit is to be run with one trapping site.
An ``rlx`` rotation with angle 0.7 radians is applied to the Na atoms followed by a 20 ms delay.
Finally the Na atom is measured.

.. parsed-literal::
    {
      'experiment_0': {
        'instructions': [
          ('rlx', [0], [0.7]),
          ('delay', [0, 1], [20]),
          ('measure', [0], []),
          ('measure', [1], [])
        ],
        'num_wires': 2,
        'shots': 10
      }
    }


The ``POST`` method of the web API will then handle this request and process it further.
In the case of the atomic mixtures backend the backend should perform the following tasks.

- Verify the provided ``access_token``.
  Users will most likely only be allowed to run jobs on the backend if they are registered and
  therefore have a valid access token.

- Assigning a unique job ID and placing the job in a job management system.
  Note that this job management is not done by ``qiskit-cold-atom``.

- Processing the circuit. This includes validation which determines if the input data corresponds
  to the outlined format and that all parameter values, including wire numbers, are within acceptable ranges.
  The input JSon data should be processed further.
  For instance, by converting it into a suitable ``experiment.py`` file for the control setup and
  running the experiment.
  The actual implementation of this is left to the backend's discretion. 
  An example of such an implementation is the `qlued <https://github.com/Alqor-UG/qlued>`_ framework.


The ``response`` of this ``POST`` request is sent back to the user as a Json that includes a ``job_id``.
This unique identification number is created by the backend for each submitted ``data`` file.
The ``job_id`` is subsequently used to define a ``Job`` object which is the central object in Qiskit
created to manage and handle the submitted task. 

Result payload
~~~~~~~~~~~~~~

To describe job results, Qiskit provides the ``Result`` class which we use without further modifications.
The Json dictionary that is returned when a user queries the backend for the result of his job can be
turned into a ``qiskit.Result`` object via the ``Result.from_dict()`` method.
A minimal configuration of the data returned by the backend is shown below.

.. parsed-literal::
    # configuration of result dictionary returned by the backend as a Json dictionary.

    {
      "backend_name": str,
      "backend_version": str,
      "job_id": str,
      "qobj_id": str,
      "success": bool,
      "header": dict,  # must be JSon serializable
      "results": list[
        {
          "header": dict,  # must be JSon serializable
          "shots": int,
          "success": bool,
          "meas_return": str,
          "meas_level": int,  # most likely always 1 or perhaps 0
          "data": {
            "counts": dict,  # must be JSon serializable
            "memory": list
          }
        }
      ]
    }

The actual information about the results of the (possibly multiple) ``QuantumCircuit`` s is
given as dictionaries themselves, which are provided as a list under the ``results`` key.
Each element in this list corresponds to one experiment (i.e. ``QuantumCircuit``).
For each individual circuit, there are two main ways the results are stored.
The default way is to store the measurement results as a dictionary under the key ``counts``.
This dictionary groups the different shots by their different measurement outcomes and
simply counts the occurrences of each outcome.
For example for a two qubit circuit with 10 shots this may look like:
``"counts": {"00": 3, "01": 1, "10": 4, "11": 2}``
This count dictionary is accessible via the ``Result.get_counts()`` method.
For the atomic mixtures we have many more degrees of freedom in the observables.

If the backend supports memory, i.e. ``"memory":True`` in the backend ``config``,
then the individual measurement outcomes of each shot are returned as a list under
the ``"memory"`` key and can be accessed through ``Result.get_memory``.
This is more appropriate for the cold atom experiments.
The ``get_memory`` function implemented in Qiskit ``Result`` can return the memory
for a specific experiment if given an index or experiment name as argument.
The format of the list returned by memory depends on the ``meas_return`` type.
If single-shots are returned the dimension of the memory is the number of shots
times the number of *memory slots*.
Each wire has one memory slot.
Each entry in the memory is specified as a list of two numbers.
For the cold atom backend the first number will represent the number of atoms in the
spin-up state while the second number will be the number of atoms in the spin down state.
When ``result.get_memory()`` is called these two numbers are returned as a single complex number.
This formatting is a result of the IQ plane description of superconducting qubits.
An example of a result is shown below.
If averaged results are returned the memory has one dimension less as the shots are averaged.

.. parsed-literal::
    # Example of a result returned by the NaLi device backend as a Json dictionary.
    # The result has one experiment (namely experiment_0 which matches the name above)
    # with three shots and two wires (one for Na and one for Li).
    # In the first shot there are 90012 Na atoms in the spin-up state and 9988 in the spin-down state.

    {
    "backend_name": "atomic_mixtures_device",
    "backend_version": "0.0.1",
    "job_id": "dae51c52-5caa-11eb-b265-080027f905c2",
    "qobj_id": None,
    "success": True,
    "header": {},
    "results": list[
        {
            "header": {"name": "experiment_0", "extra metadata": "text"},
            "shots": 3,
            "success": True,
            "meas_return": "single",
            "meas_level": 1,
            "data": {      # slot 1 (Na)      # slot 2 (Li)
                "memory": [[[90012.,  9988.], [5100., 4900.]],  # Shot 1
                           [[89900., 10100.], [5000., 5000.]],  # Shot 2
                           [[90000., 10000.], [5050., 4950.]]]  # Shot 3
            }
        }
    ]
    }


.. parsed-literal::
    # Part of the data above modified to the
    # case where average results are returned to the user.

    "shots": 3,
    "meas_return": "avg",
    "meas_level": 1,
    "data": {   # slot 1 (Na)       # slot 2 (Li)
      "memory": [[89971.,  10029.], [5050., 4950]]  # Average of three shots
    }


The ``meas_level`` and ``meas_return`` (optional) keys indicate what kind of data is returned.
Finally, depending on the backend, instead of ``counts`` or ``memory``, the dictionary of
the ``"data"`` key can also include ``statevector``, ``unitary`` or ``snapshot`` keys,
which add further flexibility to the datatypes that can be returned in a result object.
