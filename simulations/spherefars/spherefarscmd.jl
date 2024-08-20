using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using FastBEAST
using Base.Threads
using Dates

function fullmat(hmat::HMatrix{I, K}) where {I, K}

    A = zeros(K, size(hmat)[1], size(hmat)[2])
    @threads for frb in hmat.fullrankblocks
        A[frb.τ, frb.σ] = frb.M
    end

    @threads for lrb in hmat.lowrankblocks
        A[lrb.τ, lrb.σ] = lrb.M.U * lrb.M.V
    end

    return A
end

c = 3e8
μ = 4*π*1e-7
ε = 1/(μ*c^2)
f = 1e8
λ = c/f
k = 2*π/λ
ω = k*c
η = sqrt(μ/ε)

h = parse(Float64, ARGS[2])
Γ = CompScienceMeshes.meshsphere(1.0, h)

op = Maxwell3D.doublelayer(wavenumber=k)
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

if ARGS[1] == "EFIE"
    Y = raviartthomas(Γ)
    op = Maxwell3D.singlelayer(wavenumber=k)
end

piv = FastBEAST.MaxPivoting()
conv = FastBEAST.Standard()

K_bc = hassemble(
    op,
    Y,
    X,
    treeoptions=BoxTreeOptions(nmin=100),
    compressor=FastBEAST.ACAOptions(
        rowpivstrat=piv, convcrit=conv, maxrank=100, tol=1e-4
    ),    
    verbose=true,
    multithreading=true
)

A = assemble(op, Y, X)
matErr = norm(A - fullmat(K_bc))/norm(A)

lrbhmat = zeros(ComplexF64, size(A, 1), size(A, 2))
lrbmat = zeros(ComplexF64, size(A, 1), size(A, 2))
for lrb in K_bc.lowrankblocks
    lrbhmat[lrb.τ, lrb.σ] = lrb.M.U*lrb.M.V
    lrbmat[lrb.τ, lrb.σ] = A[lrb.τ, lrb.σ]
end
lrberr = norm(lrbhmat-lrbmat)/norm(lrbmat)

results = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "\n"
results = results * "sphere " * ARGS[1] * "; pivoting: MaxPivoting, convergence: Standard, h: " * ARGS[2]
results = results * "\nN \t storage in GB \t compression \t" * "rel matrix error\t" * "rel compressed matrix error\n" 
#---------------------------------------
# Write data
#---------------------------------------
file = open(pwd() * "/spherefars/results_spherefars.txt", "r")
oldresults = read(file, String)
close(file)
file = open(pwd() * "/spherefars/results_spherefars.txt", "w")
results = oldresults * results * string(length(Y.pos)) * "\t" * 
    string(FastBEAST.storage(K_bc)) * "\t" * 
    string(FastBEAST.compressionrate(K_bc)) * "\t" * 
    string(matErr) * "\t" * string(lrberr) * "\n \n"
write(file, results)
close(file)
#--------------------------------------
# Finished data
#--------------------------------------