struct KernelMatrix{
    K,
    SpaceType
} 
    fct::Function
    testspace::SpaceType
    trialspace::SpaceType

    function KernelMatrix{K}(fct, testspace, trialspace) where K

        return new{T, typeof(testspace)}(fct, testspace, trialspace)
    end
end

function KernelMatrix(fct, testspace, trialspace, K)
    return KernelMatrix{K}(fct, testspace, trialspace)
end

function (KM::KernelMatrix{K, SpaceType})(testidcs::Vector{I}, trialidcs::Vector{I}) where I
    mat = zeros(K, length(testidcs), length(trialidcs))
    for i in eachindex(testidcs)
        for j in eachindex(trialidcs)
            mat[i,j] = KM.fct(KM.testspace[testidcs[i]], KM.trialspace[trialidcs[j]])
        end
    end
    return mat
end