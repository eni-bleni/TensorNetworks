"""
    isingHamBlocks(L,J,h,g)

Return the Ising hamiltonian as a list of matrices
"""
function isingHamBlocks(L,J,h,g)
    blocks = Array{Array{ComplexF64,2},1}(undef,L-1)
    for i=1:L-1
        if i==1
            blocks[i] = -(J*ZZ + h/2*(2XI+IX) + g/2*(2*ZI+IZ))
        elseif i==L-1
            blocks[i] = -(J*ZZ + h/2*(XI+2IX) + g/2*(ZI+2*IZ))
        else
            blocks[i] = -(J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ))
        end
    end
    return blocks
end

"""
    isingHamGates(L,J,h,g)

Return the Ising hamiltonian as a list of 2-site gates
"""
function isingHamGates(L,J,h,g)
    gates = Array{Array{ComplexF64,4},1}(undef,L-1)
    for i=1:L-1
        if i==1
            gates[i] = reshape(-(J*ZZ + h/2*(2XI+IX) + g/2*(2*ZI+IZ)),2,2,2,2)
        elseif i==L-1
            gates[i] = reshape(-(J*ZZ + h/2*(XI+2IX) + g/2*(ZI+2*IZ)),2,2,2,2)
        else
            gates[i] = reshape(-(J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)),2,2,2,2)
        end
    end
    return gates
end


"""
    IdentityMPO(lattice sites, phys dims)

Return the identity MPO
"""
function IdentityMPO(L, d)
    # mpo = Array{Array{Complex{Float32},4}}(L)
    mpo = Array{Any}(L)
    for i = 1:L
        mpo[i] = Array{Complex128}(1,d,d,1)
        mpo[i][1,:,:,1] = eye(d)
    end
    return MPO(mpo)
end

"""
    translationMPO(lattice sites, matrix)

Returns a translationally invariant one-site mpo
"""
function translationMPO(L, M)
    mpo = Array{Any}(L)
    mpo[1] = Array{Complex128}(1,2,2,2)
    mpo[1][1,:,:,:] = reshape([si M],2,2,2)
    mpo[L] = Array{Complex128}(2,2,2,1)
    mpo[L][:,:,:,1] = permutedims(reshape([M si], 2,2,2), [3,1,2])
    for i=2:L-1
        # hardcoded implementation of index structure (a,i,j,b):
        help = Array{Complex128}(2,2,2,2)
        help[1,:,:,1] = help[2,:,:,2] = si
        help[1,:,:,2] = M
        help[2,:,:,1] = s0
        mpo[i] = help
    end

    return MPO(mpo)
end


"""
IsingMPO(lattice sites, J, transverse, longitudinal[, shift=0])

Returns the Ising hamiltonian as an MPO
"""
function IsingMPO(L, J, h, g, shift=0)
    mpo = Array{Array{ComplexF64,4}}(undef,L)
    mpo[1] = zeros(ComplexF64,1,2,2,3)
    mpo[1][1,:,:,:] = reshape([si J*sz h*sx+g*sz+shift*si/L],2,2,3)
    mpo[L] = zeros(ComplexF64,3,2,2,1)
    mpo[L][:,:,:,1] = permutedims(reshape([h*sx+g*sz+shift*si/L sz si], 2,2,3), [3,1,2])
    for i=2:L-1
        # hardcoded implementation of index structure (a,i,j,b):
        help = zeros(ComplexF64,3,2,2,3)
        help[1,:,:,1] = help[3,:,:,3] = si
        help[1,:,:,2] = J*sz
        help[1,:,:,3] = h*sx+g*sz+shift*si/L
        help[2,:,:,1] = help[2,:,:,2] = help[3,:,:,1] = help[3,:,:,2] = s0
        help[2,:,:,3] = sz
        mpo[i] = help
    end
    return MPO(mpo)
end

"""
    HeisenbergMPO(lattice sites,Jx,Jy,Jz,transverse)

Returns the Heisenberg gamiltonian as an MPO
"""
function HeisenbergMPO(L, Jx, Jy, Jz, h)
    mpo = Array{Any}(L)
    mpo[1] = Array{Complex128}(1,2,2,5)
    mpo[1][1,:,:,:] = reshape([si Jx*sx Jy*sy Jz*sz h*sx], 2,2,5)
    mpo[L] = Array{Complex128}(5,2,2,1)
    mpo[L][:,:,:,1] = permutedims(reshape([h*sx sx sy sz si], 2,2,5), [3,1,2])

    for i=2:L-1
        # hardcoded implementation of index structure (a,i,j,b):
        help = Array{Complex128}(5,2,2,5)
        help[1,:,:,1] = help[5,:,:,5] = si
        help[1,:,:,2] = Jx*sx
        help[1,:,:,3] = Jy*sy
        help[1,:,:,4] = Jz*sz
        help[1,:,:,5] = h*sx
        help[2,:,:,5] = sx
        help[3,:,:,5] = sy
        help[4,:,:,5] = sz
        #help[2,:,:,1:4] = help[3,:,:,1:4] = help[4,:,:,1:4] = help[5,:,:,1:4] = s0
        help[2,:,:,1] = help[2,:,:,2] = help[2,:,:,3] = help[2,:,:,4] = s0
        help[3,:,:,1] = help[3,:,:,2] = help[3,:,:,3] = help[3,:,:,4] = s0
        help[4,:,:,1] = help[4,:,:,2] = help[4,:,:,3] = help[4,:,:,4] = s0
        help[5,:,:,1] = help[5,:,:,2] = help[5,:,:,3] = help[5,:,:,4] = s0
        mpo[i] = help
    end
    return MPO(mpo)
end

"""
    TwoSiteHamToMPO(ham,L)

Returns the MPO for a 2-site Hamiltonian
"""
function TwoSiteHamToMPO(ham,L)
    d = size(ham)[1]
    mpo = Array{Any}(L)
    tmp = reshape(permutedims(ham,[1,3,2,4]),d*d,d*d)
    U,S,V = svd(tmp)
    U = reshape(U*diagm(sqrt.(S)),d,d,size(S)[1])
    V = reshape(diagm(sqrt.(S))*V',size(S)[1],d,d)
    mpo[1] = permutedims(reshape(U,d,d,size(S),1),[1,4,3,2])
    mpo[L] = permutedims(reshape(V,size(S),d,d,1),[2,1,4,3])
    @tensor begin
        tmpEven[-1,-2,-3,-4] := V[-2,-1,1]*U[1,-4,-3];
        tmpOdd[-1,-2,-3,-4] := U[-1,1,-3]*V[-2,1,-4];
    end
    for i=2:L-1
        if iseven(i)
            mpo[i] = tmpEven
        else
            mpo[i] = tmpOdd
        end
    end
    return mpo
end
