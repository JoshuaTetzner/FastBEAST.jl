using Base.Threads
using ClusterTrees
using LinearMaps
using LinearAlgebra
using ProgressMeter
using ThreadsX


struct GalerkinBlockMatrix{I, F, T} <: LinearMaps.LinearMap{F}
    self::Vector{T}
    nears::Vector{T}
    size::Tuple{I, I}
end


Base.eltype(A::GalerkinBlockMatrix{I, F, T}) where {I, F, T} = F

function Base.size(A::GalerkinBlockMatrix{I, F, T}, dim=nothing) where {I, F, T}
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
    y::AbstractVecOrMat, A::GalerkinBlockMatrix{I, F, T}, x::AbstractVector
) where {I, F, T}
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))
    
    cc = zeros(eltype(y), size(A, 1), Threads.nthreads())
    yy = zeros(eltype(y), size(A, 1), Threads.nthreads())

    @threads for mb in A.nears
        mul!(cc[1:size(mb.M, 1), Threads.threadid()], mb.M, x[mb.σ])
        yy[mb.τ, Threads.threadid()] .+= cc[1:size(mb.M,1), Threads.threadid()]
        mul!(cc[1:size(mb.M, 2), Threads.threadid()], transpose(mb.M), x[mb.τ])
        yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M,2), Threads.threadid()]
    end
    @threads for mb in A.self
        mul!(cc[1:size(mb.M, 1), Threads.threadid()], mb.M, x[mb.σ])
        yy[mb.τ, Threads.threadid()] .+= cc[1:size(mb.M,1), Threads.threadid()]
    end

    y[:] = sum(yy, dims=2)

    return y
end


@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    transA::LinearMaps.TransposeMap{<:Any,<:GalerkinBlockMatrix{I, F, T}},
    x::AbstractVector
) where {I, F, T}
    mul!(y, transA.lmap, x)
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.AdjointMap{<:Any,<:GalerkinBlockMatrix{I, F, T}},
    x::AbstractVector
) where {I, F, T} 
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    
    cc = zeros(eltype(y), size(A, 2), Threads.nthreads())
    yy = zeros(eltype(y), size(A, 2), Threads.nthreads())

    @threads for mb in A.lmap.nears
        mul!(cc[1:size(mb.M, 2), Threads.threadid()], adjoint(mb.M), x[mb.τ])
        yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M, 2), Threads.threadid()]
        mul!(cc[1:size(mb.M, 1), Threads.threadid()], transpose(adjoint(mb.M)), x[mb.σ])
        yy[mb.τ, Threads.threadid()] .+= cc[1:size(mb.M, 1), Threads.threadid()]
    end
    @threads for mb in A.lmap.self
        mul!(cc[1:size(mb.M, 2), Threads.threadid()], adjoint(mb.M), x[mb.τ])
        yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M,2), Threads.threadid()]
    end

    y[:] = sum(yy, dims=2)

    return y
end

function assemble(
    op::BEAST.AbstractOperator,
    space::BEAST.Space,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    interactions::Vector{Tuple{I, I}},
    ::Type{K};
    quadstrat=BEAST.defaultquadstrat(op, space, space),
    verbose=false,
    multithreading=true
) where {I,K,T}

    @views assembler = BEAST.blockassembler(
        op, space, space, quadstrat=quadstrat
    )
    @views function nearassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        assembler(tdata,sdata,store)
    end

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
        fullblocks[idx] = getfullmatrixview(
            nearassembler,
            value(tree.test_cluster, snear[1]),
            value(tree.trial_cluster, snear[2]),
            Int,
            scalartype(op),
        )
        verbose && next!(p)
    end
    _foreach(enumerate(selfs)) do (idx, self) 
        selfblocks[idx] = getfullmatrixview(
            nearassembler,
            value(tree.test_cluster, self[1]),
            value(tree.trial_cluster, self[2]),
            Int,
            K,
        )
        verbose && next!(p)
    end

    return GalerkinBlockMatrix{I, K, MatrixBlock{I, K, Matrix{K}}}(
        selfblocks,    
        fullblocks, 
        (tree.test_cluster.num_elements, tree.trial_cluster.num_elements)
    )
end
