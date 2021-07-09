const DEFAULT_UMPS_DMAX = 20
const DEFAULT_UMPS_TOL = 1e-12
const DEFAULT_UMPS_NORMALIZATION = true
const DEFAULT_UMPS_TRUNCATION = TruncationArgs(DEFAULT_UMPS_DMAX, DEFAULT_UMPS_TOL, DEFAULT_UMPS_NORMALIZATION)

import Base:convert

struct UMPS{T <: Number} <: AbstractMPS{T}
    #In gamma-lambda notation
    Γ::Array{Array{T,3},1}
    Λ::Array{Array{T,1},1}

    #Indicates whether the MPS should be treated as a purification or not
    purification::Bool

	# Max bond dimension and tolerance
	truncation::TruncationArgs
	#Dmax::Integer
	#tol::Float64

	# Accumulated error
	error::Base.RefValue{Float64}

    #Constructors
	function UMPS(Γ::Array{Array{T,3},1}, Λ::Array{Array{T,1},1}; truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION, purification=false, error = 0.0) where {T}
		new{T}(Γ, Λ, purification,truncation, Ref(error))
	end

end


function UMPS(Γ::Array{Array{T,3},1}, Λ::Array{Array{T,1},1}, mps::UMPS; error=0) where{T}
	return UMPS(Γ, Λ, purification = mps.purification, truncation= mps.truncation, error = mps.error[] + error)
end

function UMPS(Γ::Array{Array{T,3},1}, mps::UMPS; error=0) where {T}
	Λ = [ones(T,size(γ,1))/sqrt(size(γ,1)) for γ in Γ]
	return UMPS(Γ, Λ, purification = mps.purification, truncation= mps.truncation, error = mps.error[] + error)
end

function UMPS(Γ::Array{Array{T,3},1}, Λ::Array{Array{K,1},1}, mps::UMPS; error=0) where{T,K}
	return UMPS(Γ,map(λ->convert.(T,λ),Λ), purification = mps.purification, truncation= mps.truncation, error = mps.error[] + error)
end

Base.length(mps::UMPS) = length(mps.Γ)

"""
	convert(Type{UMPS{T}}, mps::UMPS)

Convert `mps` to an UMPS{T} and return the result
"""
function Base.convert(::Type{UMPS{T}}, mps::UMPS) where {T<:Number}
	return UMPS(map(g-> convert.(T,g),mps.Γ), map(λ-> Base.convert.(T,λ), mps.Λ), mps)
end
function Base.convert(::Type{UMPS{T}}, mps::UMPS{T}) where {T<:Number}
	return mps
end

"""
	randomUMPS(T::DataType, N, d, D; purification=false, truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION)

Return a random UMPS{T}
"""
function randomUMPS(T::DataType, N, d, D; purification=false, truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION)
	Γ = Array{Array{T,3},1}(undef,N)
	Λ = Array{Array{T,1},1}(undef,N)
	for i in 1:N
		Γ[i] = rand(T,D,d,D)
		Λ[i] = ones(T,D)
	end
	mps = UMPS(Γ,Λ, purification = purification, truncation= truncation)
	return mps
end

"""
	identityUMPS(T::DataType, N, d; truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION)

Return the UMPS corresponding to the identity density matrix
"""
function identityUMPS(T::DataType, N, d; truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION)
	Γ = Array{Array{T,3},1}(undef,N)
	Λ = Array{Array{T,1},1}(undef,N)
	for i in 1:N
		Γ[i] = reshape(Matrix{T}(I,d,d)/sqrt(d),1,d^2,1)
		Λ[i] = ones(T,1)
	end
	mps = UMPS(Γ,Λ, purification = true, truncation = truncation)
	return mps
end
function identityMPS(mps::UMPS{T}) where {T}
	N = length(mps.Γ)
	d = size(mps.Γ[1],2)
	trunc = mps.truncation
	return identityUMPS(T, N, d, truncation=trunc)
end

"""
	reverse(mps::UMPS)

Flip the spatial direction
"""
function Base.reverse(mps::UMPS)
	Γ = reverse(map(Γ->permutedims(Γ,[3,2,1]),mps.Γ))
	Λ = reverse(mps.Λ)
	return UMPS(Γ,Λ, mps)
end

"""
	rotate(mps::UMPS,n::Integer)

Rotate/shift the unit cell `n` steps
"""
function rotate(mps::UMPS,n::Integer)
	return UMPS(circshift(mps.Γ,n), circshift(mps.Λ,n), mps)
end

# %% Transfer
function transfer_matrix(mps::UMPS, op, site, direction = :left)
	oplength::Int = Int(length(size(op))/2)
	#oplength = Int(N_op/2)
	N = length(mps.Γ)
	if mps.purification
		op = auxillerate(op)
	end
    Γ::typeof(mps.Γ) = mps.Γ[mod1.(site:(site+oplength-1),N)]
	Λ::typeof(mps.Λ) = mps.Λ[mod1.(site:site+oplength,N)]
    return transfer_matrix(Γ,Λ,op,direction)
end

function transfer_matrix(mps::UMPS,site::Integer,direction=:left)
	N = length(mps.Γ)
	if direction == :left
		T = transfer_left(mps.Γ[site],mps.Λ[mod1(site+1,N)])
	elseif direction == :right
		T = transfer_right(mps.Γ[site],mps.Λ[site])
	else
		error("Choose direction :left or :right")
	end
	return T
end

function transfer_matrix_squared(mps::UMPS, site::Integer, direction=:left)
	N = length(mps.Γ)
	if direction == :left
		T = transfer_matrix_squared(mps.Γ[site], mps.Λ[mod1(site+1,N)], :left)
	elseif direction == :right
		T = transfer_matrix_squared(mps.Γ[site], mps.Λ[site], :right)
	else
		error("Choose direction :left or :right")
	end
	return T
end

"""
	transfer_spectrum(mps::UMPS, direction=:left; nev=1)

Return the spectrum of the transfer matrix of the UMPS
"""
function transfer_spectrum(mps::UMPS{K}, direction=:left; nev=1) where {K}
	if K == ComplexDF64
		@warn("converting ComplexDF64 to ComplexF64")
		mps = convert(UMPS{ComplexF64},mps)
	end
    T = transfer_matrix(mps,direction)
	D = Int(sqrt(size(T,2)))
	nev = minimum([D^2, nev])
    if size(T,1)<10
        vals, vecs = eigen(Matrix(T))
        vals = vals[end:-1:end-nev+1]
        vecs = vecs[:,end:-1:end-nev+1]
    else
        vals, vecs = eigs(T,nev=nev)
    end
	if K == ComplexDF64
		vals = ComplexDF64.(vals)
		vecs = ComplexDF64.(vecs)
	end
	# tensors = Array{Array{eltype(vecs),2},1}(undef,nev)
	# for i in 1:nev
	# 	tensors[i] = reshape(vecs[:,i],D,D)
	# end
    return vals, vecs[:,1:nev] #canonicalize_eigenoperator.(tensors)
end

function LinearAlgebra.norm(mps::UMPS)
	return sum(mps.Λ[1] .^2)
end

# %% Canonicalize
function check_canonical(mps::UMPS)

end

function canonicalize!(mps::UMPS,n)
	for i in 1:n
		apply_identity_layer!(mps,0)
		apply_identity_layer!(mps,1)
	end
	return mps
end

"""
	canonicalize_cell!(mps::UMPS)

Make the unit cell canonical
"""
function canonicalize_cell!(mps::UMPS)
	D = length(mps.Λ[1])
	N = length(mps.Γ)
	Γ = mps.Γ
	Λ = mps.Λ

	valR, rhoRs = transfer_spectrum(mps,:left,nev=2)
	valL, rhoLs = transfer_spectrum(mps,:right,nev=2)
	rhoR =  canonicalize_eigenoperator(reshape(rhoRs[:,1],D,D))
	rhoL =  canonicalize_eigenoperator(reshape(rhoLs[:,1],D,D))

    #Cholesky
	if isposdef(rhoR) && isposdef(rhoL)
    	X = Matrix(cholesky(rhoR, check=true).U)
    	Y = Matrix(cholesky(rhoL, check=true).U)
	else
		@warn("Not positive definite. Cholesky failed. Using eigen instead.")
		evl, Ul = eigen(Matrix(rhoL))
	    evr, Ur = eigen(Matrix(rhoR))
	    sevr = sqrt.(complex.(evr))
	    sevl = sqrt.(complex.(evl))
	    X = Diagonal(sevr)[abs.(sevr) .> mps.truncation.tol,:] * Ur'
	    Y = Diagonal(sevl)[abs.(sevl) .> mps.truncation.tol,:] * Ul'
	end
	F = svd(Y*Diagonal(mps.Λ[1])*transpose(X))

    #U,S,Vt,D,err = truncate_svd(F)

    #rest
    YU= pinv(Y)*F.U ./ (valL[1])^(1/4)
    VX= F.Vt*pinv(transpose(X)) ./ (valR[1])^(1/4)

    @tensor Γ[end][:] := Γ[end][-1,-2,3]*YU[3,-3]
    @tensor Γ[1][:] := VX[-1,1]*Γ[1][1,-2,-3]
	if mps.truncation.normalize
		Λ[1] = F.S ./ LinearAlgebra.norm(F.S)
	else
		Λ[1] = F.S
	end
	return
end

function canonicalize!(mps::UMPS)
	N = length(mps.Γ)
	if N>2
		error("Canonicalize with identity layers if the unit cell is larger than two sites")
	end
	canonicalize_cell!(mps)
	if N==2
		d = size(mps.Γ[1],2)
	    mps.Γ[1], mps.Λ[2], mps.Γ[2], err = apply_two_site_identity(mps.Γ, mps.Λ[mod1.(1:3,2)], mps.truncation)
	end
    return
end

function canonicalize(mps::UMPS,n)
	for i in 1:n
		mps = apply_identity_layer(mps,0)
		mps = apply_identity_layer(mps,1)
	end
	return mps
end

"""
	canonicalize_cell(mps::UMPS)

Make the unit cell canonical and return the resulting UMPS
"""
function canonicalize_cell(mps::UMPS)
	D = length(mps.Λ[1])
	N = length(mps.Γ)
	Γcopy = deepcopy(mps.Γ)
	Λcopy = deepcopy(mps.Λ)

	valR, rhoRs = transfer_spectrum(mps,:left,nev=2)
	valL, rhoLs = transfer_spectrum(mps,:right,nev=2)
	rhoR =  canonicalize_eigenoperator(reshape(rhoRs[:,1],D,D))
	rhoL =  canonicalize_eigenoperator(reshape(rhoLs[:,1],D,D))

    #Cholesky
	if isposdef(rhoR) && isposdef(rhoL)
    	X = Matrix(cholesky(rhoR, check=true).U)
    	Y = Matrix(cholesky(rhoL, check=true).U)
	else
		@warn("Not positive definite. Cholesky failed. Using eigen instead.")
		evl, Ul = eigen(Matrix(rhoL))
	    evr, Ur = eigen(Matrix(rhoR))
	    sevr = sqrt.(complex.(evr))
	    sevl = sqrt.(complex.(evl))
	    X = Diagonal(sevr)[abs.(sevr) .> mps.truncation.tol,:] * Ur'
	    Y = Diagonal(sevl)[abs.(sevl) .> mps.truncation.tol,:] * Ul'
	end
	F = svd!(Y*Diagonal(mps.Λ[1])*transpose(X))

    U,S,Vt,D,err = truncate_svd(F, mps.truncation)

    #rest
    YU= pinv(Y)*U ./ (valL[1])^(1/4)
    VX= Vt*pinv(transpose(X)) ./ (valR[1])^(1/4)

    @tensor Γcopy[end][:] := Γcopy[end][-1,-2,3]*YU[3,-3]
    @tensor Γcopy[1][:] := VX[-1,1]*Γcopy[1][1,-2,-3]
	if mps.truncation.normalize
		Λcopy[1] = S ./ LinearAlgebra.norm(S)
	else
		Λcopy[1] = S
	end
	return UMPS(Γcopy, Λcopy, mps, error = err)
end

function canonicalize(mps::UMPS)
	N = length(mps.Γ)
	#Γ = similar(mps.Γ)
	#Λ = deepcopy(mps.Λ)
	if N>2
		error("Canonicalize with identity layers if the unit cell is larger than two sites")
	end
	mps = canonicalize_cell(mps)
	if N==1
		mpsout = mps
	else N==2
		Γ = similar(mps.Γ)
		d = size(mps.Γ[1],2)
	    Γ[1],Λ2,Γ[2], err = apply_two_site_identity(mps.Γ, mps.Λ[mod1.(1:3,2)], mps.truncation)
		mpsout = UMPS(Γ,[mps.Λ[1], Λ2], mps, error = err)
	end
    return mpsout
end

function apply_mpo(mps::UMPS,mpo)
	Nmpo = length(mpo)
	Nmps = length(mps.Γ)
	N = Int(Nmps*Nmpo / gcd(Nmps,Nmpo))
	mpo = mpo[mod1.(1:N,Nmpo)]
	Γ = mps.Γ[mod1.(1:N,Nmps)]
	Λ = mps.Λ[mod1.(1:N,Nmps)]
	Γout = similar(Γ)
	Λout = similar(Λ)
	for i in 1:N
		@tensor tens[:] := Γ[i][-1,c,-4]*mpo[i][-2,-3,c,-5]
		st = size(tens)
		Γout[i] = reshape(tens,st[1]*st[2],st[3],st[4]*st[5])
		@tensor Λtemp[:] := Λ[i][-1]*ones(st[2])[-2]
		Λout[i] = reshape(Λtemp,st[1]*st[2])
	end
	return UMPS(Γout, Λout, mps)
end

# %% TEBD
"""
	double(mps::UMPS)

Return an UMPS with double the size of the unit cell
"""
function double(mps::UMPS)
	N = length(mps.Γ)
	return UMPS(mps.Γ[mod1.(1:2N,N)], mps.Λ[mod1.(1:2N,N)],mps)
end

function apply_identity_layer(mps::UMPS, shift)
	N = length(mps.Γ)
	if isodd(N)
		mps = double(mps)
		N = length(mps.Γ)
	end
	Γ = mps.Γ
	Λ = mps.Λ
	Γout = similar(Γ)
	Λout = deepcopy(Λ)
	itr = shift .+ (1:2:N-1)
	#itr = isodd(n) ? (1:2:N-1) : (s:-2:1)
	total_error = 0.0
	Threads.@threads for k in itr
		Γout[mod1(k,N)], Λout[mod1(k+1,N)], Γout[mod1(k+1,N)], error = apply_two_site_identity(Γ[mod1.(k:k+1,N)], Λ[mod1.(k:k+2, N)], mps.truncation)
		total_error+=error
	end
	return UMPS(Γout, Λout, mps, error = total_error)
end

function apply_identity_layer!(mps::UMPS, shift)
	N = length(mps.Γ)
	if isodd(N)
		error("Size of unit cell should be even to be consistent with trotter decomposition.")
	end
	Γ = mps.Γ
	Λ = mps.Λ
	itr = shift .+ (1:2:N-1)
	total_error = 0.0
	Threads.@threads for k in itr
		Γ[mod1(k,N)], Λ[mod1(k+1,N)], Γ[mod1(k+1,N)], error = apply_two_site_identity(Γ[mod1.(k:k+1,N)], Λ[mod1.(k:k+2, N)], mps.truncation)
		total_error+=error
	end
	return
end

function apply_layers!(mps::UMPS, layers)
	N = length(mps.Γ)
	if isodd(N)
		error("Cell size should be even to be consistent with trotter decomposition")
	end
	total_error = 0.0
	Nl = length(layers)
	Γ = mps.Γ
	Λ = mps.Λ
	for n in 1:Nl
		println(size.(Λ))
		layer = layers[n][mod1.((1:N) .+ (n-1), length(layers[n]))]
		total_error += apply_layer!(Γ, Λ, Γ, view(Λ, mod1.(1:N+1,N)), layer, 1, mps.truncation)
		Γ = Γ[mod1.(2:N+1,N)]
		Λ = Λ[mod1.(2:N+2,N)]
	end
	mps.Γ[1:N] = Γ[mod1.((1:N) .- Nl,N)]
	mps.Λ[1:N] = Λ[mod1.((1:N) .- Nl,N)]
	mps.error[] += total_error
	return total_error
end

function apply_layers_nonunitary!(mps::UMPS, layers)
	N = length(mps.Γ)
	if isodd(N)
		error("Cell size should be even to be consistent with trotter decomposition")
	end
	total_error = 0.0
	Nl = length(layers)
	Γ = mps.Γ
	Λ = mps.Λ
	for n=1:Nl
		if isodd(n)
			dir = 1
			Γ = @view Γ[mod1.(1:N,N)]
			Λ = @view Λ[mod1.(1:N+1,N)]
		elseif iseven(n)
			dir = -1
			Γ = @view Γ[mod1.(1:N+2,N)]
			Λ = @view Λ[mod1.(1:N+3,N)]
		end
		total_error += apply_layer_nonunitary!(Γ, Λ, Γ, Λ, layers[n], n, dir, mps.truncation)
	end
	mps.Γ[1:N] = @view Γ[1:N]
	mps.Λ[1:N] = @view Λ[1:N]
	mps.error[] += total_error
	return total_error
end

function prepare_layers(mps::UMPS, hamiltonian_gates, dt, trotter_order)
	gates = (mps.purification ? auxillerate.(hamiltonian_gates) : hamiltonian_gates)
	return prepare_layers(gates,dt,trotter_order)
end

#%% Expectation values
function expectation_value(mps::UMPS, op, site::Int)
    opDims = size(op)
	opLength=Int(length(opDims)/2)
	N = length(mps.Γ)
	if mps.purification
		op = auxillerate(op)
	end
	if opLength == 1
		val = expectation_value_one_site(mps.Λ[site],mps.Γ[site],mps.Λ[mod1(site+1,n)],op)
	elseif opLength == 2
		val = expectation_value_two_site(mps.Γ[mod1.(site:site+1,N)],mps.Λ[mod1.(site:site+2,N)],op)
	else
		error("Expectation value not implemented for operators of this size")
	end
	return val
end

function expectation_values_two_site(mps::UMPS,op)
	valRs, vecRs = transfer_spectrum(mps,:left,nev=4)
	valLs, vecLs = transfer_spectrum(mps,:right,nev=4)
	DR = length(mps.Λ[1])
	DL = length(mps.Λ[1])
	rhoRs = Matrix.(canonicalize_eigenoperator.(map(k-> reshape(vecRs[:,k],DR,DR),1:4)))
	rhoLs = Matrix.(canonicalize_eigenoperator.(map(k-> reshape(vecLs[:,k],DL,DL),1:4)))
	thetaL = similar(mps.Γ[1])
	thetaR = similar(mps.Γ[2])
    absorb_l!(thetaL,mps.Λ[1],mps.Γ[1],mps.Λ[2])
	absorb_l!(thetaR,mps.Γ[2],mps.Λ[1])
    #rs = Array{Any,2}(undef,4,4)
	f(k1,k2) = @tensoropt (d,u,r,l,Ru,Rd,Lu,Ld) rhoLs[k1][Lu,Ld]*thetaL[Ld,cld,d] *op[clu,cru,cld,crd] *conj(thetaL[Lu,clu,u]) *conj(thetaR[u,cru,Ru]) *thetaR[d,crd,Rd] *rhoRs[k2][Ru,Rd]
	rs = [ f(k1,k2) for k1 in 1:4, k2 in 1:4 ]
    return rs
end

function expectation_values(mps::UMPS,op)
    opDims = size(op)
    opLength = Int(length(opDims)/2)
    N = length(mps.Γ)
	if mps.purification
		op = auxillerate(op)
	end
	vals = Array{ComplexF64,1}(undef,N)
	for site in 1:N
		if opLength == 1
			vals[site] = expectation_value_one_site(mps.Λ[site], mps.Γ[site], mps.Λ[mod1(site+1,N)], op)
		elseif opLength == 2
			vals[site] = expectation_value_two_site(mps.Γ[mod1.(site:site+1,N)], mps.Λ[mod1.(site:site+2,N)], op)
		else
			error("Expectation value not implemented for operators of this size")
		end
	end
    return vals
end

function correlator(mps::UMPS{T},op1,op2,n) where {T}
	opsize = size(op1)
	oplength = Int(length(opsize)/2)
	transfers = transfer_matrices(mps,:right)
	N = length(transfers)
	Ts = transfers[mod1.((1+oplength):(n+oplength-1),N)]
	Rs = Vector{Transpose{T,Vector{T}}}(undef,N)
	for k in 1:N
		Tfinal = transfer_matrix(mps,op2,k, :left)
		st = Int(sqrt(size(Tfinal,2)))
		R = reshape(Matrix{T}(I,st,st),st^2)
		Rs[k] = transpose(Tfinal*R)
	end
	Tstart = transfer_matrix(mps,op1,1, :right)
	sl = Int(sqrt(size(Tstart,2)))
	L = reshape(Matrix(I,sl,sl),sl^2)
	L = Tstart*L
	vals = Array{ComplexF64,1}(undef,n-1)
	for k = 1:n-1
		Λ = Diagonal(mps.Λ[mod1.(k+oplength,N)])
		middle = kron(Λ,Λ)
		vals[k] = Rs[mod1(k+oplength,N)]*middle*L
		L = Ts[k]*L
	end
	return vals
end

"""
     canonicalize_eigenoperator(rho)

makes the dominant eigenvector hermitian
"""
function canonicalize_eigenoperator(rho)
    trρ = tr(rho)
    phase = trρ/abs(trρ)
    rho = rho ./ phase
    return  Hermitian((rho + rho')/2)
end


# %% Entropy
function renyi(mps::UMPS, n)
	N = length(mps.Γ)
	T = eltype(mps.Γ[1])
    transfer_matrices = transfer_matrices_squared(mps,:right)
	sizes = size.(mps.Γ)
	id = Matrix{T}(I,sizes[1][1],sizes[1][1])
	leftVec = vec(@tensor id[-1,-2]*id[-3,-4])
	Λsquares = diagm.(mps.Λ .^2)
	rightVecs = map(k->vec(@tensor Λsquares[-1,-2]*Λsquares[-3,-4])', 1:N)
    vals = []
    for k in 1:n
		leftVec = transfer_matrices[mod1(k,N)]*leftVec
		val = rightVecs[mod1(k+1,N)] * leftVec
        push!(vals, -log2(val))
    end
    return vals
end

function renyi(mps::UMPS)
	N = length(mps.Γ)
	T = eltype(mps.Γ[1])
    transfer_matrix = transfer_matrix_squared(mps,:right)
	return -log2(eigs(transfer_matrix,nev=1)[1])
end

# %%
function saveUMPS(mps, filename)
    jldopen(filename, "w") do file
		writeOpenMPS(file,mps)
    end
end

function writeUMPS(parent, mps)
	write(parent, "Gamma", mps.Γ)
	write(parent, "Lambda", mps.Λ)
	write(parent, "Purification", mps.purification)
	write(parent, "Dmax", mps.truncation.Dmax)
	write(parent, "tol", mps.truncation.tol)
	write(parent, "normalize", mps.truncation.normalize)
	write(parent, "error", mps.error[])
end

function readUMPS(io)
	Γ = read(io, "Gamma")
	Λ = read(io, "Lambda")
	purification = read(io, "Purification")
	Dmax = read(io, "Dmax")
	tol = read(io, "tol")
	normalize = read(io, "normalize")
	error = read(io,"error")
	trunc = TruncationArgs(Dmax, tol, normalize)
	mps = UMPS(Γ, Λ, purification = purification, truncation = trunc, error = error)
	return mps
end

function loadUMPS(filename)
	jldopen(filename, "r") do file
		global mps
		mps = readOpenMPS(file)
    end
	return mps
end
