using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using FastBEAST
using Base.Threads
using JLD2
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

h = parse(Float64, ARGS[3])
Γ = CompScienceMeshes.meshcuboid(1.0, 1.0, 1.0, h)

𝓚 = Maxwell3D.doublelayer(wavenumber=k)

X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

piv = FastBEAST.FillDistance(Y.pos)
if ARGS[1] == "thiswork"
    println("This Work")
    piv = FastBEAST.EnforcedPivoting(Y.pos)
elseif ARGS[1] == "max"
    println("Max Pivoting")
    piv = FastBEAST.MaxPivoting()
end

conv = FastBEAST.Combined(scalartype(𝓚))
if ARGS[2] == "scc"
    println("Standard Convergence")
    conv = FastBEAST.Standard()
end

K_bc = hassemble(
    𝓚,
    Y,
    X,
    treeoptions=BoxTreeOptions(nmin=100),
    compressor=FastBEAST.ACAOptions(
        rowpivstrat=piv, convcrit=conv, maxrank=100, tol=1e-4
    ),    
    verbose=true,
    multithreading=true
)

A_MFIE = assemble(𝓚, Y, X)
matErr_MFIE = norm(A_MFIE - fullmat(K_bc))/norm(A_MFIE)

results = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "\n"
results = results * "cube; pivoting: " * ARGS[1] * ", convergence: " * ARGS[2]* ", h: " * ARGS[3]
results = results * "\nN \t storage in GB \t compression \t" * "rel matrix error\n" 
#---------------------------------------
# Write data
#---------------------------------------
file = open(pwd() * "/cube/results_cube.txt", "r")
oldresults = read(file, String)
close(file)
file = open(pwd() * "/cube/results_cube.txt", "w")
results = oldresults * results * string(length(Y.pos)) * "\t" * 
    string(FastBEAST.storage(K_bc)) * "\t" * 
    string(FastBEAST.compressionrate(K_bc)) * "\t" * 
    string(matErr_MFIE) * "\n \n"
write(file, results)
close(file)
#--------------------------------------
# Finished data
#--------------------------------------