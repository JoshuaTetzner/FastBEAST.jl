using ClusterTrees
using ThreadsX

function assemble(
    op::BEAST.AbstractOperator,
    space::BEAST.Space,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    interactions::Vector{Tuple{I, I}},
    ::Type{K};
    quadstrat=BEAST.defaultquadstrat(op, space, space),
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
    blocks = Vector{BlockSparseMatrices.DenseMatrixBlock{K, Matrix{K}, Vector{I}}}(
        undef, length(snears)
    )
    diagonals = Vector{BlockSparseMatrices.DenseMatrixBlock{K, Matrix{K}, Vector{I}}}(
        undef, length(selfs)
    )

    _foreach(enumerate(snears)) do (idx, snear) 
        testidcs = value(tree.test_cluster, snear[1])
        trialidcs = value(tree.trial_cluster, snear[2])
        matrix = zeros(K, length(testidcs), length(trialidcs))
        assembler(matrix, testidcs, trialidcs)

        blocks[idx] =  BlockSparseMatrices.DenseMatrixBlock{K, Matrix{K}, Vector{Int}}(
            matrix,
            testidcs,
            trialidcs
        )
    end

    _foreach(enumerate(selfs)) do (idx, self) 
        testidcs = value(tree.test_cluster, self[1])
        trialidcs = value(tree.trial_cluster, self[2])
        matrix = zeros(K, length(testidcs), length(trialidcs))
        assembler(matrix, testidcs, trialidcs)

        diagonals[idx] =  BlockSparseMatrices.DenseMatrixBlock{K, Matrix{K}, Vector{Int}}(
            matrix,
            testidcs,
            trialidcs
        )
    end

    return SymmetricBlockMatrix(diagonals, blocks, (
        tree.test_cluster.num_elements, tree.trial_cluster.num_elements
    ))
end
