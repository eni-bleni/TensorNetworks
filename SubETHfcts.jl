using MPS
using TEBD
using TensorOperations

function subSystemThermality(state, energy ,hamBlocks, hamMpo, subSize, maxBondDim=20, steps=1000, totalTime=4)
    latticeSize = length(state)
    physD = size(state[1][2])
    rhoTherm = MPS.IdentityMPO(latticeSize,physD)
    Ethermal, betahalf = TEBD.tebd_simplified(rhoTherm,hamBlocks,totalTime,steps,maxBondDim,[],0,(energy,hamMpo))
    subTrDist = MPS.SubTraceDistance(rhoTherm,state,subSize,2)
    return subTrDist, Ethermal, 2*betahalf
end
