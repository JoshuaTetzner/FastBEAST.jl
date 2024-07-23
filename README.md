# FastBEAST
[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://FastBEAST.github.io/FastBEAST.jl/stable/)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://FastBEAST.github.io/FastBEAST.jl/dev/)
[![codecov](https://codecov.io/gh/FastBEAST/FastBEAST.jl/graph/badge.svg?token=RDRQTBWQS3)](https://codecov.io/gh/FastBEAST/FastBEAST.jl)

## Introduction
This package provides fast methods for boundary element simulations targeting [BEAST.jl](https://github.com/krcools/BEAST.jl). 

The following aspects are implemented (✔) and planned (⌛):

##### Available fast methods:
- ✔ H-matrix
- ✔ Kernel independent FMM based on [ExaFMM-t](https://github.com/exafmm/exafmm-t)
- ⌛ H2-matrix

##### Available low-rank compression techniques:
- ✔ Adaptive cross approximation 
- ⌛ Pseudo-skeleton approximation 
- ⌛ CUR matrix approximation 

## Documentation
- Documentation for the [latest stable version](https://FastBEAST.github.io/FastBEAST.jl/stable/).
- Documentation for the [development version](https://FastBEAST.github.io/FastBEAST.jl/dev/).
