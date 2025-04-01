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
