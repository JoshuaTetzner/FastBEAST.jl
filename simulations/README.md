# Simulations: On the Adaptive Cross Approximation for the Magnetic Field Integral Equation

All simulation scripts are written such that they can be called from the console from the simulation folder with the command 
```
julia --threads=XX /path/namecmd.jl argument1 argument2 argument3
```
To run all simulations for on geometry at once we provide in addition bash scripts.
We note that either the FastBEAST.jl environment must be used or the provided Project.toml be activated.

```
julia> ] dev https://github.com/FastBEAST/FastBEAST.jl.git
```