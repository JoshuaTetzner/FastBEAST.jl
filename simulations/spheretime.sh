julia --threads=16 sphere/spheretimecmd.jl thiswork ccc 0.1
julia --threads=16 sphere/spheretimecmd.jl fd ccc 0.1
julia --threads=16 sphere/spheretimecmd.jl fd scc 0.1
julia --threads=16 sphere/spheretimecmd.jl max ccc 0.1
julia --threads=16 sphere/spheretimecmd.jl max scc 0.1