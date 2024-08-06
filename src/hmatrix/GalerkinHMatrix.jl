using FLoops

struct GalerkinHMatrix{
    I, K, NearInteractionType, FarInteractionType 
} <: LinearMaps.LinearMap{K}
    nearinteractions::NearInteractionType
    farinteractions::FarInteractionType
    dim::Tuple{I, I}
    multithreading::Bool

    function GalerkinHMatrix{I, K}(
        nearinteractions,
        farinteractions,
        dim, 
        multithreading
    ) where {I, K}
        return new{I, K, typeof(nearinteractions), typeof(farinteractions)}(
            nearinteractions,
            farinteractions,
            dim,
            multithreading
        )
    end
end

function Base.size(A::GalerkinHMatrix, dim=nothing)
    if dim === nothing
        return (A.dim[1], A.dim[2])
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    else
        error("dim must be either 1 or 2")
    end
end

function GalerkinHMatrix(
    operator,
    space; 
    tree=FastBEAST.create_tree(space.pos, FastBEAST.BoxTreeoptions()),
    η=1.0,
    nearquadstrat=BEAST.defaultquadstrat(operator, space, space),
    farquadstrat=BEAST.DoubleNumQStrat(2, 2),
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    multithreading=true,
    verbose=false
)
    blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
    nears, fars = FastBEAST.computeinteractions(blktree,  η=η)
    println("nears")
    nearinteractions = FastBEAST.assemble(
        operator,
        space,
        blktree,
        nears, 
        scalartype(operator);
        quadstrat=nearquadstrat,
        verbose=verbose,
        multithreading=multithreading
    )

    fars = reduce(vcat, fars)
    sfars = eltype(fars)[]
    for far in fars
        if far[1] < far[2]
            push!(sfars, far)
        end
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    @views farasm = BEAST.blockassembler(
        operator, space, space, quadstrat=farquadstrat
    )
    @views function farassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        farasm(tdata,sdata,store)
    end
    am = FastBEAST.allocate_aca_memory(
        scalartype(operator),
        tree.num_elements,
        tree.num_elements,
        multithreading,
        maxrank=compressor.maxrank
    )
    farinteractions = Vector{FastBEAST.MatrixBlock{
        Int, scalartype(operator), FastBEAST.LowRankMatrix{scalartype(operator)}
    }}(undef, length(sfars))
    println("fars")
    _foreach(enumerate(sfars)) do (idx, far) 
        farinteractions[idx] = FastBEAST.getcompressedmatrix(
            farassembler,
            FastBEAST.value(tree, far[1]),
            FastBEAST.value(tree, far[2]),
            Int,
            scalartype(operator),
            am[Threads.threadid()],
            compressor=compressor
        )
    end

    return GalerkinHMatrix{Int, scalartype(operator)}(
        nearinteractions,
        farinteractions,
        (tree.num_elements, tree.num_elements),
        multithreading
    )
end

function assemble(operator, space; kwargs...)
    return GalerkinHMatrix(operator, space; kwargs...)
end

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::GalerkinHMatrix, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    mul!(y, A.nearinteractions, x)
    @floop for lrb in A.farinteractions
        y[lrb.τ] += lrb.M * x[lrb.σ]
    end

    return y
end
