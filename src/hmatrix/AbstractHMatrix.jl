function storage(hmat::PetrovGalerkinHMatrix)
    refsize = size(hmat, 1)*size(hmat, 2)
    thissize = 0
    for near in hmat.nearinteractions.M
        thissize += length(near.M)
    end

    for far in hmat.farinteractions.M
        thissize += length(far.M.U) + length(far.M.V)
    end

    return (thissize * 8 * 1e-6, thissize/refsize)
end

function storage(hmat::GalerkinHMatrix)
    refsize = size(hmat, 1)*size(hmat, 2)
    thissize = 0

    for near in hmat.nearinteractions.nears
        thissize += length(near.M)
    end
    
    for self in hmat.nearinteractions.self
        thissize += length(self.M)
    end


    for far in hmat.farinteractions
        thissize += length(far.M.U) + length(far.M.V)
    end

    return (thissize * 8 * 1e-9, thissize/refsize)
end

