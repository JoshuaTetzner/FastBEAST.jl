using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using FastBEAST
using JLD2
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

plate = meshrectangle(1.0,1.0, h)
plate2 = translate(plate, SVector(0,0,-0.5))
Γsrc = weld(plate, plate2)
Γtrg = translate(Γsrc, SVector(2.0, 0.0, 0.0))

𝓚 = Maxwell3D.doublelayer(wavenumber=k)

X = raviartthomas(Γsrc)
Y = buffachristiansen(Γtrg)

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

blkasm = BEAST.blockassembler(𝓚, Y, X)   
function assembler(Z, tdata, sdata)
    store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata,sdata,store)
end
lm = LazyMatrix(assembler, Vector(1:numfunctions(Y)), Vector(1:numfunctions(X)), ComplexF64)
am = allocate_aca_memory(ComplexF64, numfunctions(Y), numfunctions(X), maxrank=length(X.pos))

U, V = aca(
    lm,
    am;
    rowpivstrat=piv,
    convcrit=conv,
    tol=1e-4,
    maxrank=length(X.pos),
    svdrecompress=false
)
A_MFIE = assemble(𝓚, Y, X)
matErr_MFIE = norm(A_MFIE - U*V)/norm(A_MFIE)

results = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "\n"
results = results * "blockstructure; pivoting: "*ARGS[1]*", convergence: " * ARGS[2]*", h: " * ARGS[3] 
results = results * "\nN \t storage in MB \t compression \t" * "rel matrix error\n" 
#---------------------------------------
# Write data
#---------------------------------------
file = open(pwd() * "/blockstructure/results_blockstructure.txt", "r")
oldresults = read(file, String)
close(file)
file = open(pwd() * "/blockstructure/results_blockstructure.txt", "w")
results = oldresults * results * string(length(Y.pos)) * "\t" * 
    string(string((length(U)+length(V)) * 16 * 10^(-6))) * "\t" * 
    string((length(U)+length(V)) / length(X.pos)^2) * "\t" * 
    string(matErr_MFIE) * "\n \n"
write(file, results)
close(file)
#--------------------------------------
# Finished data
#--------------------------------------

