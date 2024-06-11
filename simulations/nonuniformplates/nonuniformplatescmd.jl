using Base.Threads
using BEAST
using CompScienceMeshes
using FastBEAST
using LinearAlgebra
using StaticArrays
using JLD2

##ihplates symmetric
src = CompScienceMeshes.read_gmsh_mesh(pwd()*"/nonuniformplates/geo/nuplates.msh")
src2 = CompScienceMeshes.rotate(src, SVector(pi, 0, 0))
trg = CompScienceMeshes.translate(src2, SVector(0, 1, -1))
Γsrc = src
Γtrg = trg

c = 3e8
f = 1e8
λ = c/f
k = 2*π/λ

MD = Maxwell3D.doublelayer(wavenumber=k)

Xsrc = raviartthomas(Γsrc)
Ytrg = buffachristiansen(Γtrg)

A = assemble(MD, Ytrg, Xsrc)

@views farblkasm = BEAST.blockassembler(
    MD,
    Ytrg,
    Xsrc,
    quadstrat=BEAST.defaultquadstrat(MD, Ytrg, Xsrc)
)

@views function farassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    farblkasm(tdata,sdata,store)
end

lm = LazyMatrix(farassembler, Vector(1:size(A, 1)), Vector(1:size(A, 2)), scalartype(MD));

conv = FastBEAST.Combined(scalartype(MD))
if ARGS[2] == "scc"
    println("Standard Convergence")
    conv = FastBEAST.Standard()
end

errors = []

for i = 1:length(Ytrg.pos)
    piv = FastBEAST.MaxPivoting(i)
    if ARGS[1] == "thiswork"
        piv = FastBEAST.EnforcedPivoting(Ytrg.pos, firstpivot=i)
    end

    U, V = aca(
        lm,
        rowpivstrat=piv, 
        convcrit=conv,
        maxrank=1000,
        tol=1e-4,
        svdrecompress=false
    );

    errorval = norm(U*V - A) / norm(A)
    push!(errors, errorval)

    println(string(errorval))
    file = open(pwd() * "/nonuniformplates/results_"*ARGS[1]*"_"*ARGS[2]*".txt", "r")
    oldresults = read(file, String)
    close(file)
    file = open(pwd() * "/nonuniformplates/results_"*ARGS[1]*"_"*ARGS[2]*".txt", "w")
    results = oldresults  * string(errorval)* "\n"
    write(file, results)
    close(file)
end

jldsave(pwd()*"/nonuniformplates/err_" * ARGS[1] * "_" * ARGS[2] * ".jld2"; errors)
