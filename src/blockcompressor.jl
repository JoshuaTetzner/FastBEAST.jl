function getcompressedmatrix(
    kernelfunction::Union{FastBEAST.KernelMatrix{K},FastBEAST.GalerkinKernelMatrix{K}},
    testidcs,
    trialidcs,
    ::Type{I},
    ::Type{K},
    am;
    compressor=ACAOptions()
) where {I, K}

    fct(mat, x, y) = kernelfunction(mat, x, y)
    lm = LazyMatrix(fct, testidcs, trialidcs, K)

    maxrank = compressor.maxrank
    maxrank == 0 && Int(round(length(lm.τ) * length(lm.σ) / (length(lm.τ) + length(lm.σ))))

    U, V = aca(
        lm,
        am;
        rowpivstrat=compressor.rowpivstrat,
        columnpivstrat=compressor.columnpivstrat,
        convcrit=compressor.convcrit,
        tol=compressor.tol,
        svdrecompress=compressor.svdrecompress,
        maxrank=maxrank
    )

    mbl = MatrixBlock{I,K,LowRankMatrix{K}}(
        LowRankMatrix(U, V),
        testidcs,
        trialidcs
    )

    return mbl
end