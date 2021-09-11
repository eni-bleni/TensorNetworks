# %% Transfer Matrices
"""
	transfer_matrices(mps::AbstractMPS, direction=:left)

Return the transfer matrices of `mps`

See also: [`transfer_matrix`](@ref), [`transfer_matrices_squared`](@ref)
"""
transfer_matrices(mps::AbstractMPS, direction=:left) = [transfer_matrix(site,direction) for site in mps[1:end]]


transfer_matrix(mps::AbstractMPS, mpo::MPOsite, site::Integer, direction=:left)	= transfer_matrix(mps[site], mpo, direction)


"""
	transfer_matrices(mps::AbstractMPS, mpo::AbstractMPO, site::Integer, direction=:left)

Return the transfer matrices of `mps` with `mpo` sandwiched

See also: [`transfer_matrix`](@ref), [`transfer_matrices_squared`](@ref)
"""
function transfer_matrices(mps::AbstractMPS, mpo::AbstractMPO, site::Integer, direction=:left)
	N = length(mpo)
	return map(k -> transfer_matrix(mps,mpo[k],site+k-1,direction), 1:N)
end

function transfer_matrix(mps::AbstractMPS,  mpo::AbstractMPO, site::Integer, direction=:left)
	Ts = transfer_matrices(mps,mpo, site, direction)
	N = length(Ts)
	if direction == :right
		Ts = Ts[N:-1:1]
	end
	return N==1 ? Ts[1] : *(Ts...)
end

"""
	transfer_matrices_squared(mps::AbstractMPS, direction=:left)

Return the transfer matrices of  the squared `mps`

See also: [`transfer_matrices`](@ref)
"""
function transfer_matrices_squared(mps::AbstractMPS, direction=:left)
	N = length(mps)
	return map(site->transfer_matrix_squared(mps, site, direction), 1:N)
end

"""
	transfer_matrix(mps::AbstractMPS, direction=:left)

Return the transfer matrix of the whole `mps`

See also: [`transfer_matrices`](@ref)
"""
function transfer_matrix(mps::AbstractMPS, direction=:left)
	Ts = transfer_matrices(mps,direction)
	N = length(Ts)
	if direction == :right
		Ts = Ts[N:-1:1]
	end
	return N==1 ? Ts[1] : *(Ts...)
end

transfer_matrices(sites::Vector{<:AbstractSite}, direction=:left) = [transfer_matrix(site, direction) for site in sites]
function transfer_matrix(sites::Vector{<:AbstractSite}, direction=:left)
	Ts = transfer_matrices(sites, direction)
	N = length(Ts)
	if direction == :right
		Ts = Ts[N:-1:1]
	end
	return N==1 ? Ts[1] : *(Ts...)
end

"""
	transfer_matrix(mps::AbstractMPS, op::AbstractGate{T,N_op}, site, direction = :left)

Return the transfer matrix at `site` with `op` sandwiched

See also: [`transfer_matrices`](@ref)
"""
function transfer_matrix(mps::AbstractMPS, op::AbstractGate{T,N_op}, site::Integer, direction = :left) where {T,N_op}
	oplength = Int(N_op/2)
    Γ = mps[site:(site+oplength-1)]
    transfer_matrix(Γ,op,direction)
end

"""
	transfer_matrix(mps::AbstractMPS, mpo::MPO, site::Integer, direction=:left)

Return the full transfer matrix with `mpo` sandwiched

See also: [`transfer_matrices`](@ref)
"""
function transfer_matrix(mps::AbstractMPS, mpo::AbstractMPO; site::Integer =1, direction=:left)
	Ts = transfer_matrices(mps,mpo,site,direction)
	N = length(Ts)
	if direction == :right
		Ts = Ts[N:-1:1]
	end
	return N==1 ? Ts[1] : *(Ts...)
end

"""
	transfer_left(Γ)

Returns the left transfer matrix of a single site

See also: [`transfer_right`](@ref)
"""
function transfer_left(Γ::Array{T,3}) where {T}
    dims = size(Γ)
    function func(Rvec)
        Rtens = reshape(Rvec,dims[3],dims[3])
        @tensoropt (t1,b1,-1,-2) temp[:] := Rtens[t1,b1]*conj(Γ[-1, c1, t1])*Γ[-2, c1, b1]
        return vec(temp)
    end
	function func_adjoint(Lvec)
		Ltens = reshape(Lvec,dims[1],dims[1])
		@tensoropt (t1,b1,-1,-2) temp[:] := Ltens[b1,t1]*conj(Γ[b1, c1, -1])*Γ[t1, c1, -2]
		return vec(temp)
	end
    return LinearMap{T}(func,func_adjoint, dims[1]^2,dims[3]^2)
end

"""
	transfer_right(Γ)

Returns the right transfer matrix of a single site

See also: [`transfer_left`](@ref)
"""
function transfer_right(Γ::Array{<:Number,3})
	Γp = reverse_direction(Γ)
	return transfer_left(Γp)
end
reverse_direction(Γ::Array{<:Number,3}) = permutedims(Γ,[3,2,1])
"""
	transfer_right(Γ, mpoSite)

Returns the right transfer matrix of a single site with `mpo` sandwiched

See also: [`transfer_left`](@ref)
"""
function transfer_right(Γ::Array{<:Number,3}, mpo::MPOsite)
	Γp = reverse_direction(Γ)
	mpop = reverse_direction(mpo)
	return transfer_left(Γp,mpop)
end

"""
	transfer_left(Γ, mpoSite)

Returns the left transfer matrix of a single site with `mpo` sandwiched

See also: [`transfer_right`](@ref)
"""
function transfer_left(Γ::Array{T,3}, mpo::MPOsite) where {T<:Number}
    dims = size(Γ)
	smpo = size(mpo)
    function func(Rvec)
        Rtens = reshape(Rvec,dims[3],smpo[4],dims[3])
        @tensoropt (tr,br,-1,-2,-3) temp[:] := conj(Γ[-1, u, tr])*mpo.data[-2,u,d,cr]*Γ[-3, d, br]*Rtens[tr,cr,br]
        return vec(temp)
    end
	function func_adjoint(Lvec)
		Ltens = reshape(Lvec,dims[1],smpo[1],dims[1])
		@tensoropt (bl,tl,-1,-2,-3) temp[:] := Ltens[bl,cl,tl]*conj(Γ[tl, u, -3])*mpo.data[cl,u,d,-2]*Γ[bl, d, -1]
		return vec(temp)
	end
    return LinearMap{T}(func,func_adjoint, smpo[1]*dims[1]^2,smpo[4]*dims[3]^2)
end



# This function gives the transfer matrix for a single site which acts on the right.
"""
	transfer_matrix_squared(A)

Return the transfer matrix for the tensor `A` squared
"""
function transfer_matrix_squared(A)
    sA=size(A)
    function contract(R)
        temp = reshape(R,sA[4],sA[4],sA[4],sA[4])
        @tensoropt (r,-2,-3,-4) begin
            temp[:] := temp[r,-2,-3,-4]*conj(A[-1,-5,-6,r])
            temp[:] := temp[-1,r,-3,-4,c,-6]*A[-2,c,-5,r]
            temp[:] := temp[-1,-2,r,-4,c,-6]*conj(A[-3,-5,c,r])
            temp[:] := temp[-1,-2,-3,r,c,-6]*A[-4,c,-5,r]
            temp[:] := temp[-1,-2,-3,-4,c,c]
        end
        st = size(temp)
        return reshape(temp,st[1]*st[2]*st[3]*st[4])
    end
    T = LinearMap{ComplexF64}(contract,sA[1]^4,sA[4]^4)
    return T
end


""" #FIXME replace A by gamma lambda
	transfer_matrix_squared(Γ::Array{T,3}, Λ::Array{T,1}, dir=:right)

Return the transfer matrix for the density matrix squared
"""
function transfer_matrix_squared(Γ::Array{T,3}, Λ::Array{T,1}, dir=:right) where {T}
	sA=size(Γ)
	d=Int(sqrt(sA[2]))
	A = reshape(A,sA[1],d,d,sA[3])
	if dir==:right
		A = permutedims(A,[4,2,3,1])
	end
	A = reshape(Λ,1,1,dims[3]) .* A
    function contract(R)
        temp = reshape(R,sA[4],sA[4],sA[4],sA[4])
        @tensoropt (r,-2,-3,-4) begin
            temp[:] := temp[r,-2,-3,-4]*A[r,-6,-5,-1]
            temp[:] := temp[-1,r,-3,-4,c,-6]*conj(A[r,-5,c,-2])
            temp[:] := temp[-1,-2,r,-4,c,-6]*A[r,c,-5,-3]
            temp[:] := temp[-1,-2,-3,r,c,-6]*conj(A[r,-5,c,-4])
            temp[:] := temp[-1,-2,-3,-4,c,c]
        end
        st = size(temp)
        return reshape(temp,st[1]*st[2]*st[3]*st[4])
    end
    return LinearMap{T}(contract,sA[1]^4,sA[4]^4)
end
