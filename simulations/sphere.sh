julia --threads=16 sphere/spherecmd.jl thiswork ccc 0.1
julia --threads=16 sphere/spherecmd.jl fd ccc 0.1
julia --threads=16 sphere/spherecmd.jl fd scc 0.1
julia --threads=16 sphere/spherecmd.jl max ccc 0.1
julia --threads=16 sphere/spherecmd.jl max scc 0.1