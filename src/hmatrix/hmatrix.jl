module HM
    using BEAST
    using ClusterTrees
    using FLoops
    using LinearMaps
    using LinearAlgebra
    using ThreadsX

    import FastBEAST
    

    include("GalerkinHMatrix.jl")
    include("PetrovGalerkinHMatrix.jl")
    include("KernelHMatrix.jl")
    include("AbstractHMatrix.jl")
    
end