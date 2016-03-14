AVX2 in Software
===============

[![Build Status](https://travis-ci.org/addaleax/sw-simd.svg?style=flat&branch=master)](https://travis-ci.org/addaleax/sw-simd?branch=master)

AVX2 software polyfill for CPUs supporting AVX instructions.

The header `sw-avx2intrin.h` provides all AVX2 intrinsics implemented on top
of AVX and SSE intrinsics. There may be notable performance losses when using
these functions, so this should only be used when testing AVX2 code
for correctness.

Possibly other intrinsics sets may be added in the future.

License
=======
GPL v3 or any later version.
