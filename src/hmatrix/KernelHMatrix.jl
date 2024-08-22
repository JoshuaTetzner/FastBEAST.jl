
function KernelHMatrix(
    kernelmatrix::FastBEAST.KernelMatrix{K}; 
    testtree=FastBEAST.create_tree(kernelmatrix.testspace, FastBEAST.BoxTreeOptions()),
    trialtree=FastBEAST.create_tree(kernelmatrix.trialspace, FastBEAST.BoxTreeOptions()),
    η=1.0,
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    multithreading=true,
    verbose=false
) where K
    blktree = ClusterTrees.BlockTrees.BlockTree(testtree, trialtree)
    nears, fars = FastBEAST.computeinteractions(blktree,  η=η)
    
    println("nears")
    @time nearinteractions = FastBEAST.assemble(
        kernelmatrix,
        blktree,
        nears;
        verbose=verbose,
        multithreading=multithreading
    )

    fars = reduce(vcat, fars)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach

    am = FastBEAST.allocate_aca_memory(
        K,
        testtree.num_elements,
        trialtree.num_elements,
        multithreading,
        maxrank=compressor.maxrank
    )
    farinteractions = Vector{FastBEAST.MatrixBlock{
        Int, K, FastBEAST.LowRankMatrix{K}
    }}(undef, length(fars))

    println("fars")
    @time _foreach(enumerate(fars)) do (idx, far) 
        farinteractions[idx] = FastBEAST.getcompressedmatrix(
            kernelmatrix,
            FastBEAST.value(testtree, far[1]),
            FastBEAST.value(trialtree, far[2]),
            Int,
            K,
            am[Threads.threadid()],
            compressor=compressor
        )
    end

    return PetrovGalerkinHMatrix{Int, K}(
        nearinteractions,
        farinteractions,
        (testtree.num_elements, trialtree.num_elements),
        multithreading
    )
end


function KernelHMatrix(
    kernelmatrix::FastBEAST.GalerkinKernelMatrix{K}; 
    tree=FastBEAST.create_tree(kernelmatrix.space, FastBEAST.BoxTreeOptions()),
    η=1.0,
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    multithreading=true,
    verbose=false
) where K
    blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
    nears, fars = FastBEAST.computeinteractions(blktree,  η=η)
    
    println("nears")
    @time nearinteractions = FastBEAST.assemble(
        kernelmatrix,
        blktree,
        nears;
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

    am = FastBEAST.allocate_aca_memory(
        K,
        tree.num_elements,
        tree.num_elements,
        multithreading,
        maxrank=compressor.maxrank
    )
    farinteractions = Vector{FastBEAST.MatrixBlock{
        Int, K, FastBEAST.LowRankMatrix{K}
    }}(undef, length(sfars))

    println("fars")
    @time _foreach(enumerate(sfars)) do (idx, far) 
        farinteractions[idx] = FastBEAST.getcompressedmatrix(
            kernelmatrix,
            FastBEAST.value(tree, far[1]),
            FastBEAST.value(tree, far[2]),
            Int,
            K,
            am[Threads.threadid()],
            compressor=compressor
        )
    end

    return GalerkinHMatrix{Int, K}(
        nearinteractions,
        farinteractions,
        (tree.num_elements, tree.num_elements),
        multithreading
    )
end
