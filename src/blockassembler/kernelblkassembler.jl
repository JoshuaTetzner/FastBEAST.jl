using Base.Threads
using ClusterTrees
using LinearMaps
using LinearAlgebra
using ProgressMeter
using ThreadsX

function assemble(
    kernelmatrix::KernelMatrix{K},
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    interactions::Vector{Tuple{I, I}};
    verbose=false,
    multithreading=true
) where {I,K,T}

    interactions=sort(interactions)
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

function assemble(
    kernelmatrix::GalerkinKernelMatrix{K},
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    interactions::Vector{Tuple{I, I}};
    verbose=false,
    multithreading=true
) where {I,K,T}

    nears = sort(interactions)
    snears = Tuple{Int, Vector{Int}}[]
    selfs = Tuple{Int, Int}[]
    for near in nears
        if near[2] > near[1]
            if length(snears)!= 0 && snears[end][1] == near[1]
                push!(snears[end][2], near[2])
            else
                push!(snears, (near[1], [near[2]]))
            end
        elseif near[1] == near[2]
            push!(selfs, near)
        end
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    fullblocks = Vector{MatrixBlock{Int,K,Matrix{K}}}(undef, length(snears))
    selfblocks = Vector{MatrixBlock{Int,K,Matrix{K}}}(undef, length(selfs))
    if verbose
        p = Progress(length(snears) + length(selfs), desc="Computing full interactions: ")
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
    _foreach(enumerate(selfs)) do (idx, self) 
        testidcs = value(tree.test_cluster, self[1])
        trialidcs = value(tree.trial_cluster, self[2])

        selfblocks[idx] = MatrixBlock{I, K, Matrix{K}}(
            kernelmatrix(testidcs, trialidcs),
            testidcs,
            trialidcs
        )
        verbose && next!(p)
    end

    return GalerkinBlockMatrix{I, K, MatrixBlock{I, K, Matrix{K}}}(
        selfblocks,    
        fullblocks, 
        (tree.test_cluster.num_elements, tree.trial_cluster.num_elements)
    )
end
