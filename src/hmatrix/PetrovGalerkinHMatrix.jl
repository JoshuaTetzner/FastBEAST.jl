struct PetrovGalerkinHMatrix{
    I, K, NearInteractionType, FarInteractionType 
} <: LinearMaps.LinearMap{K}
    nearinteractions::NearInteractionType
    farinteractions::FarInteractionType
    dim::Tuple{I, I}
    ismultithreaded::Bool

    function PetrovGalerkinHMatrix{I, K}(
        nearinteractions,
        farinteractions,
        dim, 
        ismultithreaded
    ) where {I, K}
        return new{I, K, typeof(nearinteractions), typeof(farinteractions)}(
            nearinteractions,
            farinteractions,
            dim,
            ismultithreaded
        )
    end
end

function Base.size(A::PetrovGalerkinHMatrix, dim=nothing)
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

function PetrovGalerkinHMatrix(
    operator,
    testspace,
    trialspace; 
    testtree=FastBEAST.create_tree(testspace.pos, FastBEAST.BoxTreeOptions()),
    trialtree=FastBEAST.create_tree(trialspace.pos, FastBEAST.BoxTreeOptions()),
    η=1.0,
    nearquadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    farquadstrat=BEAST.DoubleNumQStrat(2, 2),
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    multithreading=true,
    verbose=false
)
    blktree = ClusterTrees.BlockTrees.BlockTree(testtree, trialtree)
    nears, fars = FastBEAST.computeinteractions(blktree,  η=η)
    println("nears")
    @time nearinteractions = FastBEAST.assemble(
        operator,
        testspace,
        trialspace,
        blktree,
        nears, 
        scalartype(operator);
        quadstrat=nearquadstrat,
        verbose=verbose,
        multithreading=multithreading
    )

    fars = reduce(vcat, fars)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach

    @views farasm = BEAST.blockassembler(
        operator, testspace, trialspace, quadstrat=farquadstrat
    )
    @views function farassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        farasm(tdata,sdata,store)
    end
    am = FastBEAST.allocate_aca_memory(
        scalartype(operator),
        testtree.num_elements,
        trialtree.num_elements,
        multithreading,
        maxrank=compressor.maxrank
    )
    farinteractions = Vector{FastBEAST.MatrixBlock{
        Int, scalartype(operator), FastBEAST.LowRankMatrix{scalartype(operator)}
    }}(undef, length(fars))
    println("fars")
    @time _foreach(enumerate(fars)) do (idx, far) 
        farinteractions[idx] = FastBEAST.getcompressedmatrix(
            farassembler,
            FastBEAST.value(testtree, far[1]),
            FastBEAST.value(trialtree, far[2]),
            Int,
            scalartype(operator),
            am[Threads.threadid()],
            compressor=compressor
        )
    end

    return PetrovGalerkinHMatrix{Int, scalartype(operator)}(
        nearinteractions,
        farinteractions,
        (testtree.num_elements, trialtree.num_elements),
        multithreading
    )
end

function assemble(operator, testspace, trialspace; kwargs...)
    return PetrovGalerkinHMatrix(operator, testspace, trialspace; kwargs...)
end

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::PetrovGalerkinHMatrix, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    mul!(y, A.nearinteractions, x)

    _foreach = A.ismultithreaded ? ThreadsX.foreach : Base.foreach
    cc = zeros(eltype(y), size(A, 1), Threads.nthreads())
    yy = zeros(eltype(y), size(A, 1), Threads.nthreads())
    _foreach(A.farinteractions) do lrb
        mul!(cc[1:size(lrb.M, 1)], lrb.M, x[lrb.σ])
        yy[lrb.τ, Threads.threadid()] .+= cc[1:size(lrb.M, 1), Threads.threadid()]
    end

    y .+= sum(yy, dims=2)
    return y
end