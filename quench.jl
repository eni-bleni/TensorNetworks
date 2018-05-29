struct quench
    hamblock
    hamMPO
    uMPO
    operators
end

function isingQuench(L, params, operators=[])
    blocks(i, L, time) = ising_trotter_block(i,L,params,time)
    ham(time) = IsingMPO(L,params(time)...)
    uMPO(dt,time) = trotterblocks_timestep_mpo(blocks,L,dt,time)
    return quench(blocks, ham, uMPO, vcat(operators, [ham]))
end
function ising_trotter_block(i, L, params,time=0)
    J,h,g = params(time)
    if i==1
        return J*ZZ + h/2*(XI+2IX) + g/2*(ZI+2IZ)
    elseif i==L-1
        return J*ZZ + h/2*(2*XI+IX) + g/2*(2*ZI+IZ)
    else
        return J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)
    end
end

function heisenbergQuench(i,L, params,time=0)
    Jx, Jy, Jz, hx = params(time)
    XX = kron(sx, sx)
    YY = kron(sy, sy)
    ZZ = kron(sz, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)
    if i==1
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+2*IX)
    elseif i==L-1
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(2*XI+IX)
    else
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+IX)
    end
end

# magnetMPO = MPS.OneSiteMPO(latticeSize, Int(round(latticeSize/2)), [0 1; 1 0])
# magnetization = TEBD.time_evolve(mps2, ham, total_time, steps, maxBondDim, magnetMPO)
## PLOTTING
# plot(abs.(expect[:,1]), real.(expect[:,2]), show=true)



# ## thermal state MPO:
# init_params = (J0, h0, g0)
# IDmpo = MPS.IdentityMPO(latticeSize,d)
# @time TEBD.time_evolve(IDmpo,isingQuench,total_time,steps,maxBondDim,0,init_params,ETH)
# rho = MPS.multiplyMPOs(IDmpo,IDmpo)


## Ising evolution:
# init_params = (J0, h0, g0)
# println("Norm: ", MPS.MPSnorm(mps_evol))
# @time energy, entropy = TEBD.time_evolve(mps_evol,isingQuench,total_time,steps,maxBondDim,entropy_cut,init_params,ETH,"Ising")
# println("Norm: ", MPS.MPSnorm(mps_evol))
# println( "E/N = ", MPS.mpoExpectation(mps_evol,hamiltonian)/(latticeSize-1) )


## Heisenberg evolution:
# init_params = (Jx0, Jy0, Jz0, hx0)
# println("Norm: ", MPS.MPSnorm(mps_evol))
# @time energy, entropy = TEBD.time_evolve(mps_evol,heisenbergQuench,total_time,steps,maxBondDim,entropy_cut,init_params,ETH,"Heisenberg")
# println("Norm: ", MPS.MPSnorm(mps_evol))
# println( "E/N = ", MPS.mpoExpectation(mps_evol,hamiltonian)/(latticeSize-1) )


## PLOTTING
# figure(1)
# plot(abs.(energy[:,1]), real.(energy[:,2]))
# xlabel("time")
# ylabel("energy")
#
# figure(2)
# plot(abs.(entropy[:,1]), real.(entropy[:,2]))
# xlabel("time")
# ylabel("entanglement entropy")
