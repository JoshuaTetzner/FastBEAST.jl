using BEAST
using CompScienceMeshes
using FastBEAST

Γ = meshsphere(1.0, 0.08)
op = Helmholtz3D.singlelayer()
space = raviartthomas(Γ)
##
@views fassembler = BEAST.blockassembler(
    op,
    space,
    space
)
##
@views function rassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    fassembler(tdata,sdata,store)
end

x = zeros(Float64, 10, 10)
fassembler(Vector(1:10), Vector(1:10), x)
##
numfunctions(space)
##
tree = FastBEAST.create_tree(space.pos, FastBEAST.KMeansTreeOptions(nmin=50))
@time hmat = FastBEAST.HM.assemble(op, space, space, testtree=tree, trialtree=tree);
@time hmat = FastBEAST.HM.assemble(op, space, tree=tree);
##
@time hmat = FastBEAST.HM.assemble(op, space, space, testtree=tree, trialtree=tree);
@time hmat2 = hassemble(op, space, space, compressor=FastBEAST.ACAOptions(tol=1e-4), treeoptions=KMeansTreeOptions(nmin=50));
#@time A=assemble(op, space, space);
##
##
x = rand(7470)
##
@time hmat*x;
@time hmat2*x;

##

area = [1, 3, 4, 1, 5, 3]
nsamples = 6

sarea = cumsum(area)./sum(area)
rvals = rand(nsamples)

indices = zeros(Int, nsamples)
for ind in eachindex(rvals)
    indices[ind] = findfirst(x-> x > rvals[ind], sarea)
end
indices