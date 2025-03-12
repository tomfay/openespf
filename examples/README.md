### Notes on examples

Many of these examples are run using the OpenMM "OpenCL" platform. If this platform is not available to you can change platform by editing the following line:
```
platform = Platform.getPlatformByName("OpenCL")
```
to 
```
platform = Platform.getPlatformByName("CUDA")
```
if CUDA is available or
```
platform = Platform.getPlatformByName("CPU")
```
if only a CPU implement is available.
