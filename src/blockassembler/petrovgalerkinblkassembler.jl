using Base.Threads
using ClusterTrees
using LinearMaps
using LinearAlgebra
using ProgressMeter
using ThreadsX


struct BlockMatrix{I, F, T} <: LinearMaps.LinearMap{F}
    M::Vector{T}
    size::Tuple{I, I}
end


Base.eltype(A::BlockMatrix{I, F, T}) where {I, F, T} = F
function Base.size(A::BlockMatrix{I, F, T}, dim=nothing) where {I, F, T}
    if dim === nothing
        return A.size
    elseif dim == 1
        return A.size[1]
    elseif dim == 2
        return A.size[2]
    else
        error("dim must be either 1 or 2")
    end
end


@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::BlockMatrix{I, F, T}, x::AbstractVector
) where {I, F, T}
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))
    
    cc = zeros(eltype(y), size(A, 1), Threads.nthreads())
    yy = zeros(eltype(y), size(A, 1), Threads.nthreads())

    @threads for mb in A.M
        mul!(cc[1:size(mb.M, 1), Threads.threadid()], mb.M, x[mb.σ])
        yy[mb.τ, Threads.threadid()] .+= cc[1:size(mb.M,1), Threads.threadid()]
    end
        
    y[:] = sum(yy, dims=2)

    return y
end


@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    transA::LinearMaps.TransposeMap{<:Any,<:BlockMatrix{I, F, T}},
    x::AbstractVector
) where {I, F, T}
    LinearMaps.check_dim_mul(y, transA.lmap, x)

    fill!(y, zero(eltype(y)))

    cc = zeros(eltype(y), size(transA, 1), Threads.nthreads())
    yy = zeros(eltype(y), size(transA, 1), Threads.nthreads())

    @threads for mb in transA.lmap.M
        mul!(cc[1:size(mb.M, 2), Threads.threadid()], transpose(mb.M), x[mb.τ])
        yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M, 2), Threads.threadid()]
    end

    y[:] = sum(yy, dims=2)

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    transA::LinearMaps.AdjointMap{<:Any,<:BlockMatrix{I, F, T}},
    x::AbstractVector
) where {I, F, T} 
    LinearMaps.check_dim_mul(y, transA, x)

    fill!(y, zero(eltype(y)))

    cc = zeros(eltype(y), size(transA, 1), Threads.nthreads())
    yy = zeros(eltype(y), size(transA, 1), Threads.nthreads())

    @threads for mb in transA.lmap.M
        mul!(cc[1:size(mb.M, 2), Threads.threadid()], adjoint(mb.M), x[mb.τ])
        yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M, 2), Threads.threadid()]
    end
 
    y[:] = sum(yy, dims=2)

    return y
end

function assemble(
    op::BEAST.AbstractOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    interactions::Vector{Tuple{I, I}},
    ::Type{K};
    quadstrat=BEAST.defaultquadstrat(op, testspace, trialspace),
    verbose=false,
    multithreading=true
) where {I,K,T}

    @views nearblkassembler = BEAST.blockassembler(
        op, testspace, trialspace, quadstrat=quadstrat
    )
    @views function nearassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        nearblkassembler(tdata,sdata,store)
    end

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
        fullblocks[idx] = getfullmatrixview(
            nearassembler,
            value(tree.test_cluster, snear[1]),
            value(tree.trial_cluster, snear[2]),
            Int,
            K,
        )
        verbose && next!(p)
    end

    return BlockMatrix{I, K, MatrixBlock{I, K, Matrix{K}}}(fullblocks, (
        tree.test_cluster.num_elements, tree.trial_cluster.num_elements
    ))
end