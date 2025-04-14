### Installation

I'd recommend installing PySCF and OpenMM through pip and conda respectively before installing OpenESPF.

For PySCF first run:
```
pip install --prefer-binary pyscf
```
and for OpenMM:
```
conda -c conda-forge openmm
```
Then you can run:
```
pip install git+https://github.com/tomfay/openespf.git
```
If you already have openespf installed, in order to upgrade to the latest version run:
```
pip install --upgrade git+https://github.com/tomfay/openespf.git
```

### Available Features

The OpenESPF code provides a framework for performing QM/MM simulations with polarizable MM force fields within the Direct Reaction Field (DRF) framework together with the ElectroStatic Potential Fitted (ESPF) multipole operators. The OpenESPF code implements these methods and interfaces the OpenMM and PySCF codes for MM and QM calculations, although alternative MM and QM codes can in principle be switched out with relative ease. 

The following features are available:

* SCF method (DFT/HF) ESPF-DRF single point calculations and analytic energy gradients (spin-restricted, unrestricted and restricted open available).
* TD-DFT single point and analytic energy gradient calculations (spin-restricted singlet/triplet excitations only).
* General wave-function based ESPF-DRF single point calculations with incore construction and storage of electron repulsion integrals.
* Atom centered charge + dipole ESPF operators.
* Periodic boundary conditions for all methods.
* QM/MM interaction energy analysis for SCF methods.
* Pauli repulsion treated with the exchange-repulsion model.


