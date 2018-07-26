using MPS
using TEBD
using TensorOperations

# function subSystemThermality(state, energy ,hamBlocks, hamMpo,increment, maxBondDim=20, steps=1000, totalTime=-im*3)
#     lSize = length(state)
#     physD = size(state[1])[2]
#     rhoTherm = MPS.IdentityMPO(lSize,physD)
#     println("\x1b[31m Activity: \x1b[0m  search thermal state for state with energy=", energy)
#     tic()
#     Ethermal, betahalf = TEBD.tebd_simplified(rhoTherm,hamBlocks,totalTime,steps,maxBondDim,[],0,(energy,hamMpo))
#     toc()
#     println("\x1b[31m Activity: \x1b[0m compute the trace distance between a state with energy=", energy, " and the thermal state at 1/T=",2*betahalf)
#     tic()
#     subTrDist = MPS.SubTraceDistance(rhoTherm,state,subSize,2)
#     toc()
#     return subTrDist, Ethermal, 2*betahalf
# end

function save_data(data, filename; header="")
    open(filename, "a") do f
        write(f, header)
        writedlm(f, data)
        write(f,"\r\n")
    end
end


################################################################################
##                                 Initialize
################################################################################
#println("Do you want to specify the Hamiltonian? press enter and then [yes=1|no=0]")
val = 0#parse(Int64,chomp(readline()))
if val == 1
    ## input values
    println("Specify the Ising Hamiltonian:")
    println("J0 =")
    J0 = parse(Float64,readline())
    println("h0 =")
    h0 = parse(Float64,readline())
    println("g0 =")
    g0 = parse(Float64,readline())
elseif val == 0
    ## Critical Ising Hamiltonian
    J0 = 1.0
    h0 = 1.0
    g0 = 0.0
    println("Parameters are set to J0=",J0,", h0=",h0,", and g0=",g0)
else
    error("You have to type 1 or 0.")
end

latticeSize = 5
maxBondDim =5

hamblocks(time) = TEBD.isingHamBlocks(latticeSize,J0,h0,g0)
hamiltonian = MPS.IsingMPO(latticeSize, J0, h0, g0)



prec = 1e-8

totalTime = -3im
steps = 2000

NStates = 3



################################################################################
##                                  DMRG
################################################################################

## generate random (left)canonical MPS

rmps = MPS.randomMPS(latticeSize,2,maxBondDim)
MPS.makeCanonical(rmps)

## generate the first N excited states

println("Generating the first ", NStates, " states in the spectrum...")
states,energies = MPS.n_lowest_states(rmps, hamiltonian, prec, NStates)

iState = Array{Int64}(0)
lPosition = Array{Int64}(0)
subTrDist = Array{Float64}(0)

physD = size(states[1][1])[2]

for i=2:NStates
    rhoTherm = MPS.IdentityMPO(latticeSize,physD)

    println("\x1b[31m Activity: \x1b[0m  search thermal state for state with energy=", energies[i])

    tic()
    Ethermal, betahalf = TEBD.tebd_simplified(rhoTherm,hamblocks,totalTime,steps,maxBondDim,[],0,(energies[i],hamiltonian))
    toc()
    for l=1:2:latticeSize

    println("\x1b[31m Activity: \x1b[0m compute the subtrace distance between a state with energy=", energies[i], " and the thermal state at 1/T=",2*betahalf," for a subsytsem of size ", l)

    tic()
    tmp = real(MPS.SubTraceDistance(rhoTherm,states[i],l,2))
    toc()
    append!(iState,i)
    append!(lPosition,l)
    append!(subTrDist,tmp)
    end
end

save_data(iState,string(@__DIR__,"/data/subETH/subETHState.txt"))
save_data(lPosition,string(@__DIR__,"/data/subETH/subETHsub.txt"))
save_data(subTrDist,string(@__DIR__,"/data/subETH/subETHdist.txt"))
