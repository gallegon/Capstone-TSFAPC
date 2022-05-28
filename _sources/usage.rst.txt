Usage
=====

Command Line Interface (CLI)
----------------------------
Can run from the command line via

``ts_cli.py context_file``

where ``context_file`` is a path
to a JSON file containing a dictionary of string keys to values to be used as the configuration for the algorithm.

The ``tests/`` folder contains examples of context files.

Python Module
-------------
Simply ``import treesegmentation.treeseg_lib`` for access to the ``Pipeline`` object as well as the defined handler functions.

QGIS (GUI)
----------
You can install and run as a QGIS plugin from the `dev` branch on GitHub.
Support for this is limited at the moment, and the CLI or Python method is recommended.
