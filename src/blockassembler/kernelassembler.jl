using Base.Threads
using ClusterTrees
using LinearMaps
using LinearAlgebra
using ProgressMeter
using ThreadsX

function assemble(
    kernelmatrix::KernelMatrix,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    interactions::Vector{Tuple{I, I}},
    ::Type{K};
    verbose=false,
    multithreading=true
) where {I,K,T}

    snears = Tuple{Int, Vector{Int}}[]
    for near in interactions
        if length(snears)!= 0 && snears[end][1] == near[1] 
            push!(snears[end][2], near[2])
        else
            push!(snears, (near[1], [near[2]]))
        end
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    fullblocks = Vector{MatrixBlock{Int,K,Matrix{K}}}(undef, length(snears))

    if verbose
        p = Progress(length(snears), desc="Computing full interactions: ")
    end

    _foreach(enumerate(snears)) do (idx, snear) 
        testidcs = value(tree.test_cluster, snear[1])
        trialidcs = value(tree.trial_cluster, snear[2])

        fullblocks[idx] = MatrixBlock{I, K, Matrix{K}}(
            kernelmatrix(testidcs, trialidcs),
            testidcs,
            trialidcs
        )
        verbose && next!(p)
    end

    return BlockMatrix{I, K, MatrixBlock{I, K, Matrix{K}}}(fullblocks, (
        tree.test_cluster.num_elements, tree.trial_cluster.num_elements
    ))
end
