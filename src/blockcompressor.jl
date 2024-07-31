using Base.Threads
using ThreadsX

function assemblecompressedblocks(
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    interactions::Vector{Tuple{I, I}},
    assembler,
    ::Type{K};
    verbose=false,
    multithreading=true
) where {I,K,T}

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    compressedblocks = Vector{MatrixBlock{Int,K,LowRankMatrix{K}}}(undef, length(interactions))

    if verbose
        p = Progress(length(nears), desc="Computing compressable interactions: ")
    end

    am = allocate_aca_memory(
        K, 
        length(value(test_tree, 1)), 
        length(value(trial_tree, 1)), 
        multithreading; 
        maxrank=compressor.maxrank, 
    )

    _foreach(enumerate(interactions)) do (idx, interaction) 
        compressedblocks[idx] = getcompressedview(
            assembler,
            value(tree.test_cluster, interaction[1]),
            value(tree.trial_cluster, interaction[2]),
            am[Threads.threadid()],
            K,
            compressor
        )
        verbose && next!(p)
    end

    return nearblocks
end


function getcompressedview(
    matrixassembler,
    testidcs::Vector{I},
    trialidcs::Vector{I},
    ::Type{K},
    am,
    compressor::FastBEAST.ACAOptions{B, I, F}
) where {B, I, F, K}

end


function getcompressedmatrixview(
    matrixassembler,
    testidcs::Vector{I},
    trialidcs::Vector{I},
    ::Type{K},
    am,
    compressor::FastBEAST.ACAOptions{B, I, F}
) where {B, I, F, K}

    lm = LazyMatrix(matrixassembler, testidcs, trialidcs, K)

    compressor.maxrank == 0 ? compressor.maxrank = Int(
        round(length(lm.τ)*length(lm.σ)/(length(lm.τ)+length(lm.σ)))
    ) : maxrank=compressor.maxrank

    U, V, rows, cols = aca(
        lm,
        am;
        rowpivstrat=compressor.rowpivstrat,
        columnpivstrat=compressor.columnpivstrat,
        convcrit=compressor.convcrit,
        tol=compressor.tol,
        svdrecompress=compressor.svdrecompress,
        maxrank=maxrank
    )

    @views MU = U * V[:, cols]
    @views MV = U[rows, :] * V

    return MatrixBlock{I, K, ClusterMatrix{I, K}}(
        ClusterMatrix(MU, MV, rows, cols),
        testidcs,
        trialidcs
    )
end

