julia --threads=16 cube/cubesweepcmd.jl thiswork ccc 1e-5
julia --threads=16 cube/cubesweepcmd.jl fd ccc 1e-5
julia --threads=16 cube/cubesweepcmd.jl fd scc 1e-5
julia --threads=16 cube/cubesweepcmd.jl max ccc 1e-5
julia --threads=16 cube/cubesweepcmd.jl max scc 1e-5