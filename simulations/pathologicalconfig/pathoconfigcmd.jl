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
h = 0.05

##failinginteraction
Γ = meshcuboid(1.0, 1.0, 1.0, h)
𝓚 = Maxwell3D.doublelayer(wavenumber=k)
SL = Maxwell3D.singlelayer(wavenumber=k)
X = raviartthomas(translate(Γ, SVector(-0.5, 1.0, -0.25)))
Y = buffachristiansen(Γ)

τ = [456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 535,
    471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 
    487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 
    503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 
    519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534
]  
σ = [6464, 6465, 6466, 6467, 6468, 6469, 6470, 6471, 6472, 6473, 6474, 6475, 6476, 
    6477, 6478, 6479, 6480, 6481, 6482, 6483, 6484, 6485, 6486, 6487, 6488, 6489, 
    6490, 6491, 6492, 6493, 6494, 6495, 6496, 6497, 6498, 6499, 6500, 6501, 6502, 
    6503, 6504, 6505, 6506, 6507, 6508, 6509, 6510, 6511, 6512, 6513, 6514, 6515, 
    6516, 6517, 6518, 6519, 6520, 6521, 6522, 6523, 6524, 6525, 6526, 6527, 6528, 
    6529, 6530, 6531, 6532, 6533, 6534, 6535, 6536, 6537, 6538, 6539, 6540, 6541, 
    6542, 6543
]

X = BEAST.RTBasis(X.geo, X.fns[σ], X.pos[σ])
Y = BEAST.RTBasis(Y.geo, Y.fns[τ], Y.pos[τ])

blk = assemble(𝓚, Y, X)

Γ2a = meshrectangle(0.25, 0.25, 0.005)
Γ2a = translate(rotate(Γ2a, SVector(0.0, -pi/2, 0.0)), SVector(0.5, 0.25, 0.25))
Γ2b = translate(rotate(Γ2a, SVector(pi, 0.0, 0.0)), SVector(-0.5, 1.75, 0.75))
X2 = raviartthomas(Γ2b)
Y2 = buffachristiansen(Γ2a)
blk2 = assemble(𝓚, Y2, X2)

M = [
    blk2 zeros(ComplexF64, size(blk2,1), size(blk, 2));
    zeros(ComplexF64, size(blk, 1), size(blk2, 2)) blk
]
##
err = []
for i = 1:size(M, 1)
    @views function fct(B, x, y)
        B[:,:] = M[x, y]
    end

    lm = LazyMatrix(fct, Vector(1:size(M, 1)), Vector(1:size(M, 2)), ComplexF64)
    pos = [Y2.pos; Y.pos]
    U, V, r, c = aca(
        lm,
        tol=10^(-4),
        rowpivstrat=FastBEAST.EnforcedPivoting(pos, firstpivot=i),
        convcrit=FastBEAST.Combined(ComplexF64),
        svdrecompress=false
    );

    errorval = norm(U*V-M)/norm(M)
    println(string(errorval))
    file = open(pwd() * "/pathologicalconfig/results_pathological.txt", "r")
    oldresults = read(file, String)
    close(file)
    file = open(pwd() * "/pathologicalconfig/results_pathological.txt", "w")
    results = oldresults  * string(errorval)* "\n"
    write(file, results)
    close(file)
end

jldsave(pwd()*"/pathologicalconfig/err_pathological.jld2"; err)