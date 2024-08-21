
function AbstractHMatrix(
    kernelmatrix::KernelMatrix; 
    testtree=FastBEAST.create_tree(kernelmatrix.testspace, FastBEAST.BoxTreeOptions()),
    trialtree=FastBEAST.create_tree(kernelmatrix.trialspace, FastBEAST.BoxTreeOptions()),
    η=1.0,
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    multithreading=true,
    verbose=false
)
    blktree = ClusterTrees.BlockTrees.BlockTree(testtree, trialtree)
    nears, fars = FastBEAST.computeinteractions(blktree,  η=η)
    
    println("nears")
    @time nearinteractions = FastBEAST.assemble(
        kernelmatrix,
        blktree,
        nears, 
        scalartype(kernelmatrix);
        verbose=verbose,
        multithreading=multithreading
    )

    fars = reduce(vcat, fars)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach

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
            kernelmatrix,
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
