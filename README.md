# Capstone-TSFAPC
CS461 OSU Capstone Tree Segmentation from Aerial Point Clouds

GitHub repo: https://github.com/gallegon/Capstone-TSFAPC

Created by Mark G, Nicholai G, and Samuel F.

## Installation and Usage
To see full documentation of installation and usage please visit our GitHub pages
site at:
https://gallegon.github.io/Capstone-TSFAPC/

To run the algorithm up to the most completed step:
```
python ts_cli.py context_file
```

Where `context_file` is the path to the input context JSON file.
There are example JSON config files in the tests/ directory.
These files allow the user to specify parameters for the tree
segmentation algorithm.

## List of unrealized features
There are several features that we were unable to achieve during in our Capstone
timeline
- Integrating statistics processing into our script
- Full integration with PDAL pipeline
- Full integration as a QGIS plugin
- Optimizations for hierarchy building, weighted graph, and partitioning steps.

## Potential for future projects
- Further optimization of the hierachy building, weighted graph, and partitioning steps.
- Further extending the QGIS integration
- Building the algorithm as a filter and or writer for PDAL (in C++), reducing the need for python to be used for processing.  Using python as a front end to an interface (command line or GUI in QGIS) and leaving the heavy lifting of the algorithm to compiled code
- More packaging for different platforms (Mac/Linux)
