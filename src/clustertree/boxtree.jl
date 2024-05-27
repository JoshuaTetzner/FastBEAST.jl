using ClusterTrees
using LinearAlgebra

function create_tree(
    points::Vector{SVector{D, F}},
    options::BoxTreeOptions{I}
) where {D, I, F <: Real}

    hs, ct = ClusterTrees.NminTrees.boundingbox(points)
    tree = ClusterTrees.NminTrees.NminTree(length(points), center=ct, radius=F(sqrt(D))*hs)

    D == 3 ? nchildren=8 : nchildren=4
    treeoptions = ClusterTrees.NminTrees.BoxTreeOptions(dim=D)
    destination = (options.nmin, options.maxlevel)
    state = (1, ct, F(sqrt(D))*hs, 1, 1, Vector(1:length(points)), points)

    ClusterTrees.NminTrees.child!(tree, treeoptions, state, destination)

    return tree
end
