using FastBEAST
using LinearAlgebra
using LinearMaps
using StaticArrays

function OneoverR(testpoint, trialpoint)
    if testpoint == trialpoint
        return 0.0
    else
        return 1/norm(testpoint-trialpoint)
    end
end

##
N = 4000
points = [@SVector rand(3) for i = 1:N]

km = FastBEAST.KernelMatrix(OneoverR, points, points, Float64)
galerkinkm = FastBEAST.KernelMatrix(OneoverR, points, Float64)

fmat = km(Vector(1:N), Vector(1:N))
##
tree = FastBEAST.create_tree(km.testspace, FastBEAST.KMeansTreeOptions(nmin=40, nchildren=2))
hmat = FastBEAST.HM.KernelHMatrix(
    km,
    testtree=tree,
    trialtree=tree,
    compressor=FastBEAST.ACAOptions(tol=1e-5),
    multithreading=false
)
ghmat = FastBEAST.HM.KernelHMatrix(
    galerkinkm,
    tree=tree,
    compressor=FastBEAST.ACAOptions(tol=1e-5),
    multithreading=false
)
##
x = rand(size(hmat, 1))
norm(fmat*x-hmat*x)/norm(fmat*x)

norm(fmat*x-ghmat*x)/norm(fmat*x)
##
