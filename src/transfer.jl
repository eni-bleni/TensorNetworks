# %% Transfer Matrices
"""
	transfer_matrices(mps::AbstractMPS, direction=:left)

Return the transfer matrices of `mps`

See also: [`transfer_matrix`](@ref), [`transfer_matrices_squared`](@ref)
"""
function transfer_matrices(mps::AbstractMPS, direction=:left)
	N = length(mps.Γ)
	return map(site->transfer_matrix(mps,site,direction), 1:N)
end

"""
	transfer_matrices(mps::AbstractMPS, mpo::MPO, direction=:left)

Return the transfer matrices of `mps` with `mpo` sandwiched

See also: [`transfer_matrix`](@ref), [`transfer_matrices_squared`](@ref)
"""
function transfer_matrices(mps::AbstractMPS, mpo::MPO, site::Integer, direction=:left)
	N = length(mpo)
	return map(k -> transfer_matrix(mps,mpo[k],site+k-1,direction), 1:N)
end

"""
	transfer_matrices_squared(mps::AbstractMPS, direction=:left)

Return the transfer matrices of  the squared `mps`

See also: [`transfer_matrices`](@ref)
"""
function transfer_matrices_squared(mps::AbstractMPS, direction=:left)
	N = length(mps.Γ)
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

"""
	transfer_matrix(mps::AbstractMPS,site,direction=:left)

Return the transfer matrix at `site`

See also: [`transfer_matrices`](@ref)
"""
function transfer_matrix(mps::AbstractMPS,site::Integer,direction=:left)
	if direction == :left
		T = transfer_left(mps.Γ[site],mps.Λ[site+1])
	elseif direction == :right
		T = transfer_right(mps.Γ[site],mps.Λ[site])
	else
		error("Choose direction :left or :right")
	end
	return T
end

"""
	transfer_matrix(mps::OrthOpenMPS,site,direction=:left)

Return the transfer matrix at `site`

See also: [`transfer_matrices`](@ref)
"""
function transfer_matrix(mps::OrthOpenMPS,site::Integer,direction=:left)
	if direction == :left
		T = transfer_left(mps.Γ[site])
	elseif direction == :right
		T = transfer_right(mps.Γ[site])
	else
		error("Choose direction :left or :right")
	end
	return T
end

"""
	transfer_matrix(mps::AbstractMPS, op::Array{T,N_op}, site, direction = :left)

Return the transfer matrix at `site` with `op` sandwiched

See also: [`transfer_matrices`](@ref)
"""
function transfer_matrix(mps::AbstractMPS, op::Array{T,N_op}, site, direction = :left) where {T,N_op}
	#oplength::Int = Int(length(size(op))/2)
	oplength = Int(N_op/2)
	N = length(mps.Γ)
	if mps.purification
		op = auxillerate(op)
	end
    Γ = mps.Γ[site:(site+oplength-1)]
	Λ = mps.Λ[site:site+oplength]
    return transfer_matrix(Γ,Λ,op,direction)
end

"""
	transfer_matrix(mps::OrthOpenMPS, op::Array{T,N_op}, site, direction = :left)

Return the transfer matrix at `site` with `op` sandwiched

See also: [`transfer_matrices`](@ref)
"""
function transfer_matrix(mps::OrthOpenMPS, op::Array{T,N_op}, site, direction = :left) where {T,N_op}
	#oplength::Int = Int(length(size(op))/2)
	oplength = Int(N_op/2)
	N = length(mps.Γ)
	if mps.purification
		op = auxillerate(op)
	end
    Γ = mps.Γ[site:(site+oplength-1)]
    return transfer_matrix(Γ,op,direction)
end

"""
	transfer_matrix(mps::AbstractMPS, mpo::MPO, site::Integer, direction=:left)

Return the full transfer matrix with `mpo` sandwiched

See also: [`transfer_matrices`](@ref)
"""
function transfer_matrix(mps::AbstractMPS, mpo::MPO, site::Integer, direction=:left)
	Ts = transfer_matrices(mps,mpo,site,direction)
	N = length(Ts)
	if direction == :right
		Ts = Ts[N:-1:1]
	end
	return N==1 ? Ts[1] : *(Ts...)
end

function transfer_matrix(mps::OpenMPS, mpo::MPOsite, site::Integer, direction=:left)
	if mps.purification
		mpo = auxillerate(mpo)
	end
	if direction == :left
		T = transfer_left(mps.Γ[site],mps.Λ[site+1], mpo)
	elseif direction == :right
		T = transfer_right(mps.Γ[site], mps.Λ[site], mpo)
	else
		error("Choose direction :left or :right")
	end
	return T
end

function transfer_matrix(mps::OrthOpenMPS, mpo::MPOsite, site::Integer, direction=:left)
	if mps.purification
		mpo = auxillerate(mpo)
	end
	if direction == :left
		T = transfer_left(mps.Γ[site], mpo)
	elseif direction == :right
		T = transfer_right(mps.Γ[site], mpo)
	else
		error("Choose direction :left or :right")
	end
	return T
end


"""
	transfer_left(Γ, Λ)

Returns the left transfer matrix of a single site

See also: [`transfer_right`](@ref)
"""
function transfer_left(Γin::Array{T,3}, Λ) where {T<:Number}
    dims = size(Γin)
    Γ = reshape(Λ,1,1,dims[3]) .* Γin
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
	transfer_left(Γ)

Returns the left transfer matrix of a single site

See also: [`transfer_right`](@ref)
"""
function transfer_left(Γin::Array{T,3}) where {T<:Number}
    dims = size(Γin)
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
	transfer_right(Γ, Λ)

Returns the right transfer matrix of a single site

See also: [`transfer_left`](@ref)
"""
function transfer_right(Γ::Array{T,3}, Λ) where {T}
	Γp = permutedims(Γ,[3,2,1])
	return transfer_left(Γp,Λ)
end
"""
	transfer_right(Γ)

Returns the right transfer matrix of a single site

See also: [`transfer_left`](@ref)
"""
function transfer_right(Γ::Array{T,3}) where {T}
	Γp = permutedims(Γ,[3,2,1])
	return transfer_left(Γp)
end

"""
	transfer_right(Γ, Λ, mpoSite)

Returns the right transfer matrix of a single site with `mpo` sandwiched

See also: [`transfer_left`](@ref)
"""
function transfer_right(Γ::Array{T,3},Λ, mpo::MPOsite) where {T}
	Γp = permutedims(Γ,[3,2,1])
	mpop = MPOsite(permutedims(mpo,[4,2,3,1]))
	return transfer_left(Γp,Λ,mpop)
end
"""
	transfer_right(Γ, mpoSite)

Returns the right transfer matrix of a single site with `mpo` sandwiched

See also: [`transfer_left`](@ref)
"""
function transfer_right(Γ::Array{T,3}, mpo::MPOsite) where {T}
	Γp = permutedims(Γ,[3,2,1])
	mpop = MPOsite(permutedims(mpo,[4,2,3,1]))
	return transfer_left(Γp,mpop)
end

"""
	transfer_left(Γ, Λ, mpoSite)

Returns the left transfer matrix of a single site with `mpo` sandwiched

See also: [`transfer_right`](@ref)
"""
function transfer_left(Γin::Array{T,3},Λ, mpo::MPOsite) where {T}
    dims = size(Γin)
	smpo = size(mpo)
    Γ =  Γin .* reshape(Λ,1,1,dims[3])
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


function transfer_matrix(Γ::Array{Array{T,3},1}, Λ, op::Array{T_op,N_op}, direction = :left) where {T, T_op<:Number, N_op}
	#oplength::Int = Int(length(opsize)/2)
	oplength = Int(N_op/2)
	opsize = size(op)
	if direction == :left
		Γnew = deepcopy(reverse(Γ))
		for k = 1:oplength
			 absorb_l!(Γnew[oplength+1-k] ,Γ[k], Λ[k+1])
			 Γnew[oplength+1-k] = permutedims(Γnew[oplength+1-k], [3,2,1])
			 op = permutedims(op,[oplength:-1:1..., 2*oplength:-1:oplength+1...])
			 #Γnew[k] = absorb_l(Γ[k], Λ[mod1(k+1,N)], :right)
		end
	elseif direction == :right
		Γnew = deepcopy(Γ)
		for k = 1:oplength
			 absorb_l!(Γnew[k],Λ[k],Γ[k])
		end
	else
		error("Specify :left or :right in transfer matrix calculation")
	end

	#println([(2*(1:oplength) .- 1)..., 2*(1:oplength)...])
	op = reshape(permutedims(op,[(2*(1:oplength) .- 1)..., 2*(1:oplength)...]), *(opsize...))
	s_start = size(Γnew[1],1)
	s_final = size(Γnew[oplength],3)

	function T_on_vec(vec)
		v = reshape(vec,1,s_start,s_start)
		for k in 1:oplength
			@tensoropt (1,2) v[:] := conj(Γnew[k][1,-2,-4])* v[-1,1,2]* Γnew[k][2,-3,-5]
			sv = size(v)
			v = reshape(v,*(sv[1:3]...),sv[4],sv[5])
		end
		@tensor v[:] := v[1,-1,-2] * op[1]
		#v = reshape(ncon((v,op),[[1:2*oplength...,-1,-2],1:2*oplength]),s_final^2)
		return reshape(v,s_final^2)
	end
	return LinearMap{ComplexF64}(T_on_vec,s_final^2,s_start^2)
end

function transfer_matrix(Γ::Array{Array{T,3},1}, op::Array{T_op,N_op}, direction = :left) where {T, T_op<:Number, N_op}
	#oplength::Int = Int(length(opsize)/2)
	oplength = Int(N_op/2)
	opsize = size(op)
	if direction == :left
		Γnew = deepcopy(reverse(Γ))
		for k = 1:oplength
			 Γnew[oplength+1-k] = permutedims(Γnew[oplength+1-k], [3,2,1])
			 op = permutedims(op,[oplength:-1:1..., 2*oplength:-1:oplength+1...])
			 #Γnew[k] = absorb_l(Γ[k], Λ[mod1(k+1,N)], :right)
		end
	elseif direction == :right
		Γnew = copy(Γ)
	else
		error("Specify :left or :right in transfer matrix calculation")
	end

	#println([(2*(1:oplength) .- 1)..., 2*(1:oplength)...])
	op = reshape(permutedims(op,[(2*(1:oplength) .- 1)..., 2*(1:oplength)...]), *(opsize...))
	s_start = size(Γnew[1],1)
	s_final = size(Γnew[oplength],3)

	function T_on_vec(vec)
		v = reshape(vec,1,s_start,s_start)
		for k in 1:oplength
			@tensoropt (1,2) v[:] := conj(Γnew[k][1,-2,-4])* v[-1,1,2]* Γnew[k][2,-3,-5]
			sv = size(v)
			v = reshape(v,*(sv[1:3]...),sv[4],sv[5])
		end
		@tensor v[:] := v[1,-1,-2] * op[1]
		#v = reshape(ncon((v,op),[[1:2*oplength...,-1,-2],1:2*oplength]),s_final^2)
		return reshape(v,s_final^2)
	end
	return LinearMap{ComplexF64}(T_on_vec,s_final^2,s_start^2)
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
