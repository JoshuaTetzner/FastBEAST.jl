using ThreadsX
using ProgressMeter

function assemblefullblocks(
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    interactions::Vector{Tuple{I, I}},
    assembler,
    ::Type{K};
    verbose=false,
    multithreading=true
) where {I,K,T}

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    fullblocks = Vector{MatrixBlock{Int,K,Matrix{K}}}(undef, length(interactions))

    if verbose
        p = Progress(length(interactions), desc="Computing full interactions: ")
    end

    _foreach(enumerate(interactions)) do (idx, interaction) 
        fullblocks[idx] = getfullmatrixview(
            assembler,
            value(tree.test_cluster, interaction[1]),
            value(tree.trial_cluster, interaction[2]),
            Int,
            K,
        )
        verbose && next!(p)
    end

    return fullblocks
end

function getfullmatrixview(
    matrixassembler,
    testidcs::Vector{I},
    trialidcs::Vector{I},
    ::Type{I},
    ::Type{K};
) where {I, K}
    matrix = zeros(K, length(testidcs), length(trialidcs))
    matrixassembler(matrix, testidcs, trialidcs)

    return MatrixBlock{I, K, Matrix{K}}(
        matrix,
        testidcs,
        trialidcs
    )
end