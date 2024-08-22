struct KernelMatrix{
    K,
    SpaceType
} 
    fct::Function
    testspace::SpaceType
    trialspace::SpaceType

    function KernelMatrix{K}(fct, testspace, trialspace) where K

        return new{K, typeof(testspace)}(fct, testspace, trialspace)
    end
end

function KernelMatrix(fct, testspace, trialspace, K)
    return KernelMatrix{K}(fct, testspace, trialspace)
end

function (KM::KernelMatrix{K})(testidcs::Vector{I}, trialidcs::Vector{I}) where {I, K}
    mat = zeros(K, length(testidcs), length(trialidcs))
    for i in eachindex(testidcs)
        for j in eachindex(trialidcs)
            mat[i,j] = KM.fct(KM.testspace[testidcs[i]], KM.trialspace[trialidcs[j]])
        end
    end
    return mat
end

function (KM::KernelMatrix{K})(mat::AbstractArray{K}, testidcs::AbstractArray{I}, trialidcs::AbstractArray{I}) where {I, K}
    for i in eachindex(testidcs)
        for j in eachindex(trialidcs)
            mat[i,j] = KM.fct(KM.testspace[testidcs[i]], KM.trialspace[trialidcs[j]])
        end
    end
    return mat
end

# GalerkinKernelMatrix

struct GalerkinKernelMatrix{
    K,
    SpaceType
} 
    fct::Function
    space::SpaceType

    function GalerkinKernelMatrix{K}(fct, space) where K

        return new{K, typeof(space)}(fct, space)
    end
end

function KernelMatrix(fct, space, K)
    return GalerkinKernelMatrix{K}(fct, space)
end

function (KM::GalerkinKernelMatrix{K})(testidcs::Vector{I}, trialidcs::Vector{I}) where {I, K}
    mat = zeros(K, length(testidcs), length(trialidcs))
    for i in eachindex(testidcs)
        for j in eachindex(trialidcs)
            mat[i,j] = KM.fct(KM.space[testidcs[i]], KM.space[trialidcs[j]])
        end
    end
    return mat
end

function (KM::GalerkinKernelMatrix{K})(
    mat::AbstractArray{K}, testidcs::AbstractArray{I}, trialidcs::AbstractArray{I}
) where {I, K}
    for i in eachindex(testidcs)
        for j in eachindex(trialidcs)
            mat[i,j] = KM.fct(KM.space[testidcs[i]], KM.space[trialidcs[j]])
        end
    end
    return mat
end