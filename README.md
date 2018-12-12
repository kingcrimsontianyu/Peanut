# What is Peanut
Utility code for Nvidia CUDA-enabled GPUs. An important component is implementation of atomic-add methods with high efficiency and low numerical errors. This component is primarily intended for the tally (i.e. scoring) functions in GPU-accelerated Monte Carlo code.

# Publication
GPU-Specific Atomic-Add Tally Methods With High Efficiency and Small Numerical Errors For Monte Carlo Radiation Dose Calculation (to be submitted).

# How to use Peanut
Include the following header file in your device code:

```
src/atomic/include/PeanutAtomic.h
```

GPU architecture | Best method to use
---------------- | -------------
Kepler/Maxwell       | double Peanut::atomicAddFP64WAG(double* address, double val)
                     | float2 Peanut::atomicAddKahanOptimized(float2* address, float val)
Pascal/Volta/Turing  | double atomicAdd(double* address, double val)
                     | double Peanut::atomicAddFP64WIB(double* address, double val)


# How to build and test Peanut
Peanut is written in C++/CUDA, using Cmake as the build and test system.

On Linux, to configure:
```
cmake \
-DCMAKE_CXX_COMPILER=/usr/local/bin/g++-4.9.4 \
-DCMAKE_C_COMPILER=/usr/local/bin/gcc-4.9.4 \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_TESTING=ON \
../Peanut
```

Then to build:
```
make test-atomic
```

Then to test:
```
ctest -V -R test_gpu_atomic_TitanXPascal
```
All tests are listed in `unit_test/atomic/CMakeLists.txt`.



