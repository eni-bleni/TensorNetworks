# %% Transfer Matrices
"""
	transfer_left(Γ)

Returns the left transfer matrix of a single site

See also: [`transfer_right`](@ref)
"""
# function transfer_left(Γ::Array{T,3}) where {T}
#     dims = size(Γ)
#     function func(Rvec)
#         Rtens = reshape(Rvec,dims[3],dims[3])
#         @tensoropt (t1,b1,-1,-2) temp[:] := Rtens[t1,b1]*conj(Γ[-1, c1, t1])*Γ[-2, c1, b1]
#         return vec(temp)
#     end
# 	function func_adjoint(Lvec)
# 		Ltens = reshape(Lvec,dims[1],dims[1])
# 		@tensoropt (t1,b1,-1,-2) temp[:] := Ltens[t1,b1]*Γ[t1, c1, -1]*conj(Γ[b1, c1, -2])
# 		return vec(temp)
# 	end
#     return LinearMap{T}(func,func_adjoint, dims[1]^2,dims[3]^2)
# end
# transfer_left(Γ::Array{<:Number,3}) = transfer_left(Γ, Γ)
# function transfer_left(Γ1::Array{T,3}, Γ2::Array{T,3}) where {T}
#     dims1 = size(Γ1)
# 	dims2 = size(Γ2)
#     function func(Rvec)
#         Rtens = reshape(Rvec,dims1[3],dims2[3])
#         @tensoropt (t1,b1,-1,-2) temp[:] := Rtens[t1,b1]*conj(Γ1[-1, c1, t1])*Γ2[-2, c1, b1]
#         return vec(temp)
#     end
# 	function func_adjoint(Lvec)
# 		Ltens = reshape(Lvec,dims1[1],dims2[1])
# 		@tensoropt (t1,b1,-1,-2) temp[:] := Ltens[t1,b1]*Γ1[t1, c1, -1]*conj(Γ2[b1, c1, -2])
# 		return vec(temp)
# 	end
#     return LinearMap{T}(func,func_adjoint, dims1[1]*dims2[1],dims1[3]*dims2[3])
# end


function transfer_left(mposites::Vararg{MPOsite,N}) where {N}
	#sizes = size.(mposites)
	rs = size.(mposites,4)
	ls = size.(mposites,1)
	#ds = size.(mposites,3)
	us = size.(mposites,2)
    function contract(R)
		site = data(mposites[1])
		# temp = reshape(R, rs[1], prod(rs[2:N])) #0.014480 seconds (250 allocations: 3.682 MiB)
		# @tensor temp[newdone, remaining, down, hat] := site[newdone,hat,down,rc] * temp[rc,remaining]

		# temp = reshape(R, rs[1], prod(rs[2:N])) 0.013839 seconds (160 allocations: 3.674 MiB)
		# @tensor temp[down, remaining, newdone, hat] := site[newdone,hat,down,rc] * temp[rc,remaining]

		temp = reshape(R, rs[1], prod(rs[2:N]))
		@tensor temp[down, remaining, hat, newdone] := site[newdone,hat,down,rc] * temp[rc,remaining]
		for k in 2:N
			site = data(mposites[k])

			temp = reshape(temp, us[k], rs[k], prod(rs[k+1:N]), us[1], prod(ls[1:k-1]))
			@tensor temp[down, remaining, hat, done,newdone] := site[newdone,upc,down,rc] * temp[upc, rc,remaining,hat,done] order=(upc,rc)

			# temp = reshape(temp, us[k], rs[k], prod(rs[k+1:N]), prod(ls[1:k-1]), us[1])  0.013839 seconds (160 allocations: 3.674 MiB)
			# @tensor temp[down, remaining, done,newdone,hat] := site[newdone,upc,down,rc] * temp[upc, rc,remaining,done,hat] order=(upc,rc)

			# temp = reshape(temp, prod(ls[1:k-1]), rs[k], prod(rs[k+1:N]), us[k], us[1])  #0.014480 seconds (250 allocations: 3.682 MiB)
			# @tensor temp[done, newdone, remaining, down, hat] := site[newdone,upc,down,rc] * temp[done,rc,remaining,upc,hat] order=(rc,upc)
		end
		if us[1] != 1
			@tensor temp[:] := temp[1,-1, 1,-2,-3]
		end
        return temp = reshape(temp, prod(ls))
    end
	function adjoint_contract(R)
		temp = reshape(R, ls[1], prod(ls[2:N]))
		site = permutedims(conj(mposites[1]),[4,2,3,1])
		@tensor temp[down, remaining, hat, newdone] := site[newdone,hat,down,rc] * temp[rc,remaining]
		for k in 2:N
			site = permutedims(conj(mposites[k]),[4,2,3,1])

			temp = reshape(temp, us[k], ls[k], prod(ls[k+1:N]), us[1], prod(rs[1:k-1]))
			@tensor temp[down, remaining, hat, done,newdone] := site[newdone,upc,down,rc] * temp[upc, rc,remaining,hat,done] order=(upc,rc)
		end
		if us[1] != 1
			@tensor temp[:] := temp[1,-1, 1,-2,-3]
		end
        return temp = reshape(temp, prod(rs))
    end

    map = LinearMap{ComplexF64}(contract, adjoint_contract, prod(ls), prod(rs))
    return map
end

transfer_right(mposites::Vararg{MPOsite}) = transfer_left(reverse_direction.(mposites)...)
reverse_direction(Γ::Array{<:Number,3}) = permutedims(Γ,[3,2,1])

# """
# 	transfer_left(Γ, AbstractMPOsite)

# Returns the left transfer matrix of a single site with `mpo` sandwiched

# See also: [`transfer_right`](@ref)
# """
# function transfer_left(Γ1::Array{T,3}, mpo::AbstractMPOsite, Γ2::Array{T,3}) where {T<:Number}
#     dims1 = size(Γ1)
# 	dims2 = size(Γ2)
# 	smpo = size(mpo)
#     function func(Rvec)
#         Rtens = reshape(Rvec,dims1[3],smpo[4],dims2[3])
#         @tensoropt (tr,br,-1,-2,-3) temp[:] := conj(Γ1[-1, u, tr])*data(mpo)[-2,u,d,cr]*Γ2[-3, d, br]*Rtens[tr,cr,br]
#         return vec(temp)
#     end
# 	function func_adjoint(Lvec)
# 		Ltens = reshape(Lvec,dims1[1],smpo[1],dims2[1])
# 		@tensoropt (bl,tl,-1,-2,-3) temp[:] := Ltens[tl,cl,bl]*Γ1[tl, u, -1]*conj(data(mpo)[cl,u,d,-2])*conj(Γ2[bl, d, -3])
# 		return vec(temp)
# 	end
#     return LinearMap{T}(func,func_adjoint, smpo[1]*dims1[1]*dims2[1],smpo[4]*dims1[3]*dims2[3])
# end
# function transfer_left(Γ::Array{T,3}, mpo::AbstractMPOsite) where {T<:Number}
#     dims = size(Γ)
# 	smpo = size(mpo)
#     function func(Rvec)
#         Rtens = reshape(Rvec,dims[3],smpo[4],dims[3])
#         @tensoropt (tr,br,-1,-2,-3) temp[:] := conj(Γ[-1, u, tr])*mpo.data[-2,u,d,cr]*Γ[-3, d, br]*Rtens[tr,cr,br]
#         return vec(temp)
#     end
# 	function func_adjoint(Lvec)
# 		Ltens = reshape(Lvec,dims[1],smpo[1],dims[1])
# 		@tensoropt (bl,tl,-1,-2,-3) temp[:] := Ltens[tl,cl,bl]*Γ[tl, u, -1]*conj(mpo.data[cl,u,d,-2])*conj(Γ[bl, d, -3])
# 		return vec(temp)
# 	end
#     return LinearMap{T}(func,func_adjoint, smpo[1]*dims[1]^2,smpo[4]*dims[3]^2)
# end

# transfer_left(Γ1::Array{<:Number,3}, mpo::ScaledIdentityMPOsite, Γ2::Array{<:Number,3}) = data(mpo) * transfer_left(Γ1, Γ2)
# transfer_right(Γ1::Array{<:Number,3}, mpo::ScaledIdentityMPOsite, Γ2::Array{<:Number,3}) = data(mpo) * transfer_right(Γ1, Γ2)

# """
# 	transfer_right(Γ)

# Returns the right transfer matrix of a single site

# See also: [`transfer_left`](@ref)
# """
# function transfer_right(Γ1::Array{<:Number,3},Γ2::Array{<:Number,3})
# 	Γ1p = reverse_direction(Γ1)
# 	Γ2p = reverse_direction(Γ2)
# 	return transfer_left(Γ1p,Γ2p)
# end
# transfer_right(Γ::Array{<:Number,3}) = transfer_left(reverse_direction(Γ))



# """
# 	transfer_right(Γ, AbstractMPOsite)

# Returns the right transfer matrix of a single site with `mpo` sandwiched

# See also: [`transfer_left`](@ref)
# """
# function transfer_right(Γ1::Array{<:Number,3}, mpo::AbstractMPOsite, Γ2::Array{<:Number,3})
# 	Γ1p = reverse_direction(Γ1)
# 	Γ2p = reverse_direction(Γ2)
# 	mpop = reverse_direction(mpo)
# 	return transfer_left(Γ1p,mpop,Γ2p)
# end
# transfer_right(Γ::Array{<:Number,3}, mpo::AbstractMPOsite) = transfer_right(Γ, mpo, Γ)

# transfer_right(Γ1::Tuple{Array{<:Number,3}}, op::AbstractMPOsite, Γ2::Tuple{Array{<:Number,3}}) = transfer_right(Γ1[1],op,Γ2[1])
# transfer_left(Γ1::Tuple{Array{<:Number,3}}, op::AbstractMPOsite, Γ2::Tuple{Array{<:Number,3}}) = transfer_left(Γ1[1],op,Γ2[1])


transfer_left(Γ1::Array{<:Number,3},  g::ScaledIdentityGate{<:Number,2}, Γ2::Array{<:Number,3}) = data(g)*transfer_left(Γ1,Γ2)
transfer_right(Γ1::Array{<:Number,3},  g::ScaledIdentityGate{<:Number,2}, Γ2::Array{<:Number,3}) = data(g)*transfer_right(Γ1,Γ2)

function transfer_left(Γ1::NTuple{N, Array{<:Number,3}}, g::ScaledIdentityGate{<:Number,N2}, Γ2::NTuple{N, Array{<:Number,3}}) where {N,N2} 
	@assert 2*N==N2
	Ts = [transfer_left(Γ1[k],Γ2[k]) for k in 1:N]
	return data(g)*(N == 1 ? Ts[1] : *(Ts...))
end
function transfer_right(Γ1::NTuple{N, Array{<:Number,3}}, g::ScaledIdentityGate{<:Number,N2}, Γ2::NTuple{N, Array{<:Number,3}}) where {N,N2} 
	@assert 2*N==N2
	Ts = [transfer_right(Γ1[k],Γ2[k]) for k in N:-1:1]
	return data(g)*(N == 1 ? Ts[1] : *(Ts...))
end


transfer_right(Γ::NTuple{N, Array{<:Number,3}}, gate::AbstractSquareGate) where {N} = transfer_right_gate(Γ,gate)
transfer_right(Γ1::NTuple{N, Array{<:Number,3}}, gate::AbstractSquareGate, Γ2::NTuple{N,Array{<:Number,3}}) where {N} = transfer_right_gate(Γ1,gate,Γ2)

function transfer_left(Γ1::NTuple{N,Array{<:Number,3}}, gate::AbstractSquareGate, Γ2::NTuple{N,Array{<:Number,3}}) where {N} 
	oplength = length(gate)
	Γnew1 = copy(reverse([Γ1...]))
	Γnew2 = copy(reverse([Γ2...]))
	for k = 1:oplength
		Γnew1[oplength+1-k] = reverse_direction(Γnew1[oplength+1-k])
		Γnew2[oplength+1-k] = reverse_direction(Γnew2[oplength+1-k])
		gate = permutedims(gate,[oplength:-1:1..., 2*oplength:-1:oplength+1...])
	end
	return transfer_right_gate(Tuple(Γnew1), gate, Tuple(Γnew2))
end 

function transfer_left(Γ::NTuple{N,Array{<:Number,3}}, gate::AbstractSquareGate) where {N} 
	oplength = length(gate)
	Γnew = copy(reverse([Γ...]))
	for k = 1:oplength
		Γnew[oplength+1-k] = reverse_direction(Γnew[oplength+1-k])
		gate = permutedims(gate,[oplength:-1:1..., 2*oplength:-1:oplength+1...])
	end
	return transfer_right_gate(Tuple(Γnew), gate)
end 


function transfer_right_gate(Γ1::NTuple{N, Array{<:Number,3}}, gate::GenericSquareGate, Γ2::NTuple{N, Array{<:Number,3}}) where {N}
	op = data(gate)
    oplength = length(gate)
	@assert length(Γ1) == oplength == length(Γ2) "Error in transfer_right_gate: number of sites does not match gate length"
	@assert size(gate,1) == size(Γ1[1],2) == size(Γ2[1],2) "Error in transfer_right_gate: physical dimension of gate and site do not match"
	perm = [Int(floor((k+1)/2))+ oplength*iseven(k) for k in 1:2*oplength]
	opvec = vec(permutedims(op,perm))
	s_start1 = size(Γ1[1])[1]
	s_start2 = size(Γ2[1])[1]
	s_final1 = size(Γ1[oplength])[3]
	s_final2 = size(Γ2[oplength])[3]
	function T_on_vec(invec)
		v = reshape(invec,1,s_start1,s_start2)
		for k in 1:oplength
			@tensoropt (1,2) v[:] := Γ1[k][1,-2,-4]* v[-1,1,2]* Γ2[k][2,-3,-5]
			sv = size(v)
			v = reshape(v,*(sv[1:3]...),sv[4],sv[5])
		end
		@tensor v[:] := v[1,-1,-2] * opvec[1]
		return vec(v)
	end
	#TODO Define adjoint
	return LinearMap{eltype(Γ1[1])}(T_on_vec,s_final1*s_final2,s_start1*s_start2)
end
function transfer_right_gate(Γ::NTuple{N, Array{<:Number,3}}, gate::GenericSquareGate) where {N}
	op = data(gate)
    oplength = length(gate)
	@assert length(Γ) == oplength "Error in transfer_right_gate: number of sites does not match gate length"
	@assert size(gate,1) == size(Γ[1],2) "Error in transfer_right_gate: physical dimension of gate and site do not match"
	perm = [Int(floor((k+1)/2))+ oplength*iseven(k) for k in 1:2*oplength]
	opvec = vec(permutedims(op,perm))
	s_start = size(Γ[1])[1]
	s_final = size(Γ[oplength])[3]
	function T_on_vec(invec)
		v = reshape(invec,1,s_start,s_start)
		for k in 1:oplength
			@tensoropt (1,2) v[:] := conj(Γ[k][1,-2,-4])* v[-1,1,2]* Γ[k][2,-3,-5]
			sv = size(v)
			v = reshape(v,*(sv[1:3]...),sv[4],sv[5])
		end
		@tensor v[:] := v[1,-1,-2] * opvec[1]
		return vec(v)
	end
	#TODO Define adjoint
	return LinearMap{eltype(Γ[1])}(T_on_vec,s_final^2,s_start^2)
end

#Sites 
transfer_matrix(site::AbstractSite,dir::Symbol =:left) = transfer_matrix((site',site),dir)
transfer_matrix(site1::AbstractSite,site2::AbstractSite,dir::Symbol =:left) = transfer_matrix((site1,site2),dir)
transfer_matrix(site::AbstractSite,op::AbstractMPOsite, dir::Symbol =:left) = transfer_matrix((site',op,site),dir)
transfer_matrix(site1::AbstractSite,op::AbstractMPOsite,site2::AbstractSite, dir::Symbol =:left) = transfer_matrix((site1,op,site2),dir)
transfer_matrix(site::AbstractSite,op::ScaledIdentityGate, dir::Symbol =:left) = data(op)*transfer_matrix((site',site),dir)
transfer_matrix(site1::AbstractSite,op::ScaledIdentityGate,site2::AbstractSite, dir::Symbol =:left) = data(op)*transfer_matrix((site1',site2),dir)

function transfer_matrix(sites::NTuple{N,Union{<:AbstractMPOsite, <:AbstractSite}}, direction::Symbol=:left) where {N}
	# sites = [sitestuple...]
	newsites = MPOsite[]
	scaling = I
	purify = ispurification(sites[1])
	for site in sites
		if typeof(site) <: ScaledIdentityMPOsite
			scaling *= data(site)
		elseif typeof(site) <: AbstractSite
			push!(newsites, MPOsite(site, reverse_direction(direction)))
		else
			if purify 
				push!(newsites, auxillerate(site))
			else
				push!(newsites, site)
			end
		end
	end
	if direction == :left
		T = transfer_left(newsites...)
	elseif direction == :right
		T = transfer_right(newsites...)
	else
		error("Choose direction :left or :right")
	end
	return scaling*T
end

# transfer_matrix(site1::AbstractSite, op::AbstractSquareGate, site2::AbstractSite, direction::Symbol=:left) = transfer_matrix(tuple(site1'),op,tuple(site2),direction)
transfer_matrix(site1::ConjugateSite{<:AbstractSite}, op::GenericSquareGate, site2::AbstractSite, direction::Symbol=:left) = transfer_matrix(tuple(site1),op,tuple(site2),direction)
function transfer_matrix(site1::NTuple{N, ConjugateSite{<:AbstractSite}}, op::AbstractSquareGate, site2::NTuple{N, <:AbstractSite}, direction::Symbol=:left) where {N}
	@assert length(site1) == length(site2) == length(op)
	if ispurification(site1[1])
		@assert ispurification(site2[1])
		op = auxillerate(op)
	end
	s1 = data.(site1, reverse_direction(direction))
	s2 = data.(site2, reverse_direction(direction))
	if direction == :left
		T = transfer_left(s1, op, s2)
	elseif direction == :right
		T = transfer_right(s1, op, s2)
	else
		error("Choose direction :left or :right")
	end
	return T
end
function transfer_matrix(site1::NTuple{N, ConjugateSite{<:GenericSite}}, op::ScaledIdentityGate, site2::NTuple{N, <:GenericSite}, direction::Symbol=:left) where {N}
	@assert length(site1) == length(site2) == length(op)
	transfer_matrix([site1...], ScaledIdentityMPO(data(op),N),[site2...], direction)
end
# transfer_matrix(site1::NTuple{N, <:GenericSite}, op::Matrix{<:Number}, site2::NTuple{N, <:GenericSite}, direction::Symbol=:left) where {N} = transfer_matrix(site1, Gate(op), site2, direction)

transfer_matrix(site::AbstractSite, op::AbstractSquareGate{<:Number,2}; direction::Symbol=:left) = transfer_matrix(tuple(site'),op,tuple(site), direction)
transfer_matrix(sites::NTuple{N, <:AbstractSite}, op::AbstractSquareGate{<:Number,N2}; direction::Symbol=:left) where {N,N2} = transfer_matrix(adjoint.(sites),op,sites, direction)
#transfer_matrix(site1::AbstractSite, site2::AbstractSite; direction::Symbol=:left) = transfer_matrix(tuple(site1), IdentityGate(1), tuple(site2), direction = direction)
#transfer_matrix(site::AbstractSite; direction::Symbol=:left) = transfer_matrix(tuple(site), IdentityGate(1),tuple(site), direction = direction)

# function transfer_matrix(site1::NTuple{N,OrthogonalLinkSite}, op::AbstractSquareGate, site2::NTuple{N,OrthogonalLinkSite}, direction::Symbol=:left) where {N}
# 	Γ1 = GenericSite.(site1, reverse_direction(direction))
# 	Γ2 = GenericSite.(site2, reverse_direction(direction))
# 	transfer_matrix(Γ1, op, Γ2, direction)
# end

##Vector of sites
function transfer_matrices(sites1::Vector{<:AbstractSite}, op::Union{AbstractMPOsite, AbstractSquareGate}, sites2::Vector{<:AbstractSite}, direction::Symbol=:left) 
	@assert length(sites1) == length(sites2) 
	n = length(op)
	[transfer_matrix(Tuple(sites1[k:k+n-1]), op, Tuple(sites2[k:k+n-1]), direction) for k in 1:length(sites1)+1-n]
end

function transfer_matrices(sites1::Vector{<:AbstractSite}, op::Vector{<:AbstractSquareGate}, sites2::Vector{<:AbstractSite}, direction::Symbol=:left)
	@assert length(sites1) == length(sites2) == length(op)
	N = length(sites1)
	Ts = []
	for k in 1:N
		n = length(op[k])
		if k+n-1>N
			break
		end
		Tm = transfer_matrix(Tuple(sites1[k:k+n-1]), op[k], Tuple(sites2[k:k+n-1]), direction)
		push!(Ts,Tm)
	end
	Ts
end
function transfer_matrices(sites1::Vector{<:AbstractSite}, op::Union{AbstractMPO,Vector{<:MPOsite}}, sites2::Vector{<:AbstractSite}, direction::Symbol=:left)
	@assert length(sites1) == length(sites2) == length(op)
	N = length(sites1)
	Ts = []
	for k in 1:N
		Tm = transfer_matrix((sites1[k], op[k], sites2[k]), direction)
		push!(Ts,Tm)
	end
	Ts
end
transfer_matrices(sites1::Vector{<:AbstractSite}, op::Vector{<:Matrix}, sites2::Vector{<:AbstractSite}, direction::Symbol=:left) = transfer_matrices(sites1, MPO(op), sites2, direction)

transfer_matrices(sites1::Vector{<:AbstractSite}, sites2::Vector{<:AbstractSite}, direction::Symbol=:left) = transfer_matrices(sites1, IdentityMPOsite, sites2, direction)

transfer_matrices(sites::Vector{<:AbstractSite},op, direction::Symbol=:left) = transfer_matrices(adjoint.(sites),op,sites,direction)
transfer_matrices(sites::Vector{<:AbstractSite}, direction::Symbol=:left) = transfer_matrices(adjoint.(sites),sites,direction)


function transfer_matrix(sites1::Vector{<:AbstractSite}, op, sites2::Vector{<:AbstractSite}, direction::Symbol=:left)
	Ts = transfer_matrices(sites1, op,sites2, direction)
	N = length(Ts)
	if direction == :right
		Ts = Ts[N:-1:1]
	end
	return N==1 ? Ts[1] : *(Ts...)
end
transfer_matrix(sites1::Vector{<:AbstractSite}, sites2::Vector{<:AbstractSite}, direction::Symbol=:left) = transfer_matrix(sites1, IdentityMPOsite, sites2, direction)

transfer_matrix(sites::Vector{<:AbstractSite},op, direction::Symbol=:left) = transfer_matrix(adjoint.(sites),op,sites,direction)
transfer_matrix(sites::Vector{<:AbstractSite}, direction::Symbol=:left) = transfer_matrix(adjoint.(sites),sites,direction)


"""
	transfer_matrices(mps::AbstractMPS, direction=:left)

Return the transfer matrices of `mps`

See also: [`transfer_matrix`](@ref), [`transfer_matrices_squared`](@ref)
"""
# transfer_matrices(mps::AbstractMPS, direction=:left) = [transfer_matrix(site,direction) for site in mps[1:end]]

transfer_matrices(mps1::AbstractMPS, op, mps2::AbstractMPS, direction::Symbol=:left) = transfer_matrices(mps1[1:end], op, mps2[1:end], direction)
transfer_matrices(mps1::AbstractMPS, mps2::AbstractMPS, direction::Symbol=:left) = transfer_matrices(mps1, IdentityMPOsite, mps2, direction)
transfer_matrices(mps::AbstractMPS, op, direction=:left) = transfer_matrices(mps', op, mps, direction)
transfer_matrices(mps::AbstractMPS, direction::Symbol=:left) = transfer_matrices(mps',IdentityMPOsite, mps, direction)


# """
# 	transfer_matrices_squared(mps::AbstractMPS, direction=:left)

# Return the transfer matrices of  the squared `mps`

# See also: [`transfer_matrices`](@ref)
# """
# function transfer_matrices_squared(mps::AbstractMPS, direction=:left)
# 	N = length(mps)
# 	return map(site->transfer_matrix_squared(mps, site, direction), 1:N)
# end


"""
	transfer_matrix(mps::AbstractMPS, op::AbstractGate{T,N_op}, site, direction = :left)

Return the transfer matrix at `site` with `op` sandwiched

See also: [`transfer_matrices`](@ref)
"""
# function transfer_matrix(mps::AbstractMPS, op::AbstractGate{T,N_op}, site::Integer, direction = :left) where {T,N_op}
# 	oplength = Int(N_op/2)
#     Γ = mps[site:(site+oplength-1)]
#     transfer_matrix(Γ,op,direction)
# end
transfer_matrix(mps::AbstractMPS, op::AbstractGate{T,N_op}, site::Integer, direction = :left) where {T,N_op} = transfer_matrix(mps' ,op ,mps, site, direction)
function transfer_matrix(mps::AbstractMPS, op::AbstractGate{T,N_op}, mps2::AbstractMPS, site::Integer, direction = :left) where {T,N_op}
	oplength = Int(N_op/2)
    Γ1 = mps[site:(site+oplength-1)]
	Γ2 = mps2[site:(site+oplength-1)]
    transfer_matrix(Γ1,op,Γ2,direction)
end

"""
	transfer_matrix(mps::AbstractMPS, mpo::AbstractMPO, site::Integer, direction=:left)

Return the full transfer matrix with `mpo` sandwiched

See also: [`transfer_matrices`](@ref)
"""
# function transfer_matrix(mps::AbstractMPS, mpo::AbstractMPO; site::Integer =1, direction=:left)
# 	Ts = transfer_matrices(mps,mpo,site,direction)
# 	N = length(Ts)
# 	if direction == :right
# 		Ts = Ts[N:-1:1]
# 	end
# 	return N==1 ? Ts[1] : *(Ts...)
# end
transfer_matrix(mps::AbstractMPS, mpo::AbstractMPO, direction::Symbol=:left) = transfer_matrix(mps',mpo,mps,direction)
transfer_matrix(mps::AbstractMPS, direction::Symbol=:left) = transfer_matrix(mps',mps,direction)
transfer_matrix(mps1::AbstractMPS, mps2::AbstractMPS, direction::Symbol=:left) = transfer_matrix(mps1,IdentityMPO(length(mps1)), mps2,direction)
# transfer_matrix(mps::AbstractMPS, mpo::AbstractMPO, site::Integer, direction=:left) = transfer_matrix(mps,mpo,mps,site,direction)
function transfer_matrix(mps1::AbstractMPS, op, mps2::AbstractMPS, direction::Symbol=:left)
	return transfer_matrix(mps1[1:end],op,mps2[1:end], direction)
	# N = length(Ts)
	# if direction == :right
	# 	Ts = Ts[N:-1:1]
	# end
	# return N==1 ? Ts[1] : *(Ts...)
end





# This function gives the transfer matrix for a single site which acts on the right.
# """
# 	transfer_matrix_squared(A)

# Return the transfer matrix for the tensor `A` squared
# """
# function transfer_matrix_squared(A)
#     sA=size(A)
#     function contract(R)
#         temp = reshape(R,sA[4],sA[4],sA[4],sA[4])
#         @tensoropt (r,-2,-3,-4) begin
#             temp[:] := temp[r,-2,-3,-4]*conj(A[-1,-5,-6,r])
#             temp[:] := temp[-1,r,-3,-4,c,-6]*A[-2,c,-5,r]
#             temp[:] := temp[-1,-2,r,-4,c,-6]*conj(A[-3,-5,c,r])
#             temp[:] := temp[-1,-2,-3,r,c,-6]*A[-4,c,-5,r]
#             temp[:] := temp[-1,-2,-3,-4,c,c]
#         end
#         st = size(temp)
#         return reshape(temp,st[1]*st[2]*st[3]*st[4])
#     end
#     T = LinearMap{ComplexF64}(contract,sA[1]^4,sA[4]^4)
#     return T
# end


# """ #FIXME replace A by gamma lambda
# 	transfer_matrix_squared(Γ::Array{T,3}, Λ::Array{T,1}, dir=:right)

# Return the transfer matrix for the density matrix squared
# """
# function transfer_matrix_squared(Γ::Array{T,3}, Λ::Array{T,1}, dir=:right) where {T}
# 	sA=size(Γ)
# 	d=Int(sqrt(sA[2]))
# 	A = reshape(A,sA[1],d,d,sA[3])
# 	if dir==:right
# 		A = permutedims(A,[4,2,3,1])
# 	end
# 	A = reshape(Λ,1,1,dims[3]) .* A
#     function contract(R)
#         temp = reshape(R,sA[4],sA[4],sA[4],sA[4])
#         @tensoropt (r,-2,-3,-4) begin
#             temp[:] := temp[r,-2,-3,-4]*A[r,-6,-5,-1]
#             temp[:] := temp[-1,r,-3,-4,c,-6]*conj(A[r,-5,c,-2])
#             temp[:] := temp[-1,-2,r,-4,c,-6]*A[r,c,-5,-3]
#             temp[:] := temp[-1,-2,-3,r,c,-6]*conj(A[r,-5,c,-4])
#             temp[:] := temp[-1,-2,-3,-4,c,c]
#         end
#         st = size(temp)
#         return reshape(temp,st[1]*st[2]*st[3]*st[4])
#     end
#     return LinearMap{T}(contract,sA[1]^4,sA[4]^4)
# end
