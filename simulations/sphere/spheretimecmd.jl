using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using FastBEAST
using Dates

c = 3e8
μ = 4*π*1e-7
ε = 1/(μ*c^2)
f = 1e8
λ = c/f
k = 2*π/λ
ω = k*c
η = sqrt(μ/ε)

h = parse(Float64, ARGS[3])
Γ = CompScienceMeshes.meshsphere(1.0, h)

pc = CompScienceMeshes.meshsphere(1.0, 0.4)

𝓚 = Maxwell3D.doublelayer(wavenumber=k)

X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

Xpc = raviartthomas(pc)
Ypc = buffachristiansen(pc)

piv = FastBEAST.FillDistance(Y.pos)
if ARGS[1] == "thiswork"
    println("This Work")
    piv = FastBEAST.EnforcedPivoting(Y.pos)
elseif ARGS[1] == "max"
    println("Max Pivoting")
    piv = FastBEAST.MaxPivoting()
end

pivpc = FastBEAST.FillDistance(Ypc.pos)
if ARGS[1] == "thiswork"
    println("This Work")
    pivpc = FastBEAST.EnforcedPivoting(Ypc.pos)
elseif ARGS[1] == "max"
    println("Max Pivoting")
    pivpc = FastBEAST.MaxPivoting()
end

conv = FastBEAST.Combined(scalartype(𝓚))
if ARGS[2] == "scc"
    println("Standard Convergence")
    conv = FastBEAST.Standard()
end

hassemble(
    𝓚,
    Ypc,
    Xpc,
    treeoptions=BoxTreeOptions(nmin=5),
    quadstratcbk=BEAST.DoubleNumQStrat(2,2),
    compressor=FastBEAST.ACAOptions(
        rowpivstrat=pivpc, convcrit=conv, maxrank=100, tol=1e-4
    ),    
    verbose=true,
    multithreading=true
)

comptime = @elapsed K_bc = hassemble(
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

results = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "\n"
results = results * "sphere; pivoting: " * ARGS[1] * ", convergence: " * ARGS[2] * ", h: " * ARGS[3]
results = results * "\nN \t time \t storage in GB \t compression \n" 
#---------------------------------------
# Write data
#---------------------------------------
file = open(pwd() * "/sphere/results_spheretime.txt", "r")
oldresults = read(file, String)
close(file)
file = open(pwd() * "/sphere/results_spheretime.txt", "w")
results = oldresults * results * string(length(Y.pos)) * "\t" * 
    string(comptime) * "\t" * string(FastBEAST.storage(K_bc)) * "\t" * 
    string(FastBEAST.compressionrate(K_bc)) * "\n \n"
write(file, results)
close(file)
#--------------------------------------
# Finished data
#--------------------------------------