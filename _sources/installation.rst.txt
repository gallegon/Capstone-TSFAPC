Installation
============

Pre-requisites
--------------
The following package requires both PDAL and GDAL.  The easiest way to 
get both of these in a single environment is to use the OSGeo4w shell.

Install OSGeo4w shell or QGIS Standalone version 3.24 or greater you can find
the download here: https://qgis.org/en/site/forusers/download.html
QGIS comes installed with the OSGeo4w

The treesegmentation package has the following
python dependencies:

pip (for building)

NumPy version 1.22.3 or greater

SciPy

GDAL

PDAL

Installation
------------

There is an included wheel file in the git repository. This is the easiest way to
install all of the required dependencies.  It is recommended to install the wheel
from the command line in the OSGeo4w shell--this will handle PDAL/GDAL dependencies
properly.  From the directory that contains the wheel file use the command:

.. code-block:: console

    $ pip install treesegmentation-0.0.1-py3-none-any.whl

Alternatively the packages can be installed from pip with the follow commands:

.. code-block:: console

    $ pip install numpy
    $ pip install scipy
    $ pip install gdal
    $ pip install pdal

Building from source
--------------------

To build from source the treesegmentation package requires build
which can be installed with pip:

.. code-block:: console

    $ pip install build

Then run:

.. code-block:: console

    $ python -m build

This will build a wheel that can be installed with pip in the
/dist/ directory.