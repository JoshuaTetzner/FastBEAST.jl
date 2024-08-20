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

h = parse(Float64, ARGS[1])
Γ = CompScienceMeshes.meshsphere(1.0, h)

T = Maxwell3D.singlelayer(wavenumber=k)

X = raviartthomas(Γ)

piv = FastBEAST.MaxPivoting()
conv = FastBEAST.Standard()

T_bc = hassemble(
    T,
    X,
    X,
    treeoptions=BoxTreeOptions(nmin=100),
    compressor=FastBEAST.ACAOptions(
        rowpivstrat=piv, convcrit=conv, maxrank=100, tol=1e-4
    ),    
    verbose=true,
    multithreading=true
)

A_EFIE = assemble(T, X, X)
matErr_MFIE = norm(A_EFIE - fullmat(T_bc))/norm(A_EFIE)

results = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "\n"
results = results * "sphere EFIE; pivoting: MaxPivoting, convergence: Standard, h: " * ARGS[1]
results = results * "\nN \t storage in GB \t compression \t" * "rel matrix error\n" 
#---------------------------------------
# Write data
#---------------------------------------
file = open(pwd() * "/sphere/results_sphereEFIE.txt", "r")
oldresults = read(file, String)
close(file)
file = open(pwd() * "/sphere/results_sphereEFIE.txt", "w")
results = oldresults * results * string(length(Y.pos)) * "\t" * 
    string(FastBEAST.storage(K_bc)) * "\t" * 
    string(FastBEAST.compressionrate(K_bc)) * "\t" * 
    string(matErr_MFIE) * "\n \n"
write(file, results)
close(file)
#--------------------------------------
# Finished data
#--------------------------------------