========================================
Sampling weights of deep neural networks
========================================

This repository contains the source code for the `paper`_ E. Bolager, I. Burak, C. Datar, Q. Sun, F. Dietrich. Sampling weights of deep neural networks. arXiv:2306.16830, 2023.

Structure
------------

**swimnetworks/** implements the SWIM algorithm for sampling weights of neural networks.
You can find the source code for four numerical experiments considered in the paper under **experiments/**. For more details on the experiments, please take a look at the *README.rst* files in the corresponding subfolders.

Installation
------------

To install the main package with the requirements, one needs to clone the repository and execute the following command from the root folder:

.. code-block:: bash

    pip install .

Each subfolder of **experiments/** contains a separate *requirements.txt* file with additional dependencies required to reproduce a specific experiment.

Up-to-date version of SWIM
--------------------------
**swimnetworks/** contains the source code that was used to run the experiments, and thus it will not be updated. You can find the latest version of the package in `this repository <https://gitlab.com/felix.dietrich/swimnetworks/>`__.

.. _paper: https://arxiv.org/abs/2306.16830
