const DEFAULT_UMPS_DMAX = 20
const DEFAULT_UMPS_TOL = 1e-12
const DEFAULT_UMPS_NORMALIZATION = true
const DEFAULT_UMPS_TRUNCATION = TruncationArgs(DEFAULT_UMPS_DMAX, DEFAULT_UMPS_TOL, DEFAULT_UMPS_NORMALIZATION)
isinfinite(::UMPS) = true
Base.firstindex(mps::UMPS) = 1
Base.lastindex(mps::UMPS) = length(mps.Γ)
Base.IndexStyle(::Type{UMPS}) = IndexLinear()
function Base.getindex(mps::UMPS, i::Integer) 
	i1 = mod1(i, length(mps))
	i2 = mod1(i+1, length(mps))
	return OrthogonalLinkSite(mps.Γ[i1], mps.Λ[i1], mps.Λ[i2])
end

function Base.setindex!(mps::UMPS, v::OrthogonalLinkSite, i::Integer)
	i1 = mod1(i, length(mps))
	i2 = mod1(i+1, length(mps))
	mps.Γ[i1] =  v.Γ
	mps.Λ[i1] =  v.Λ1
	mps.Λ[i2] =  v.Λ2
end

# %% Constructors
function UMPS(Γ::Vector{Array{T,3}}, Λ::Vector{Vector{K}}; truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION, error = 0.0, purification=false) where {T,K}
	UMPS{T}(GenericSite.(Γ,purification), LinkSite.(Λ), truncation, error)
end
function UMPS(Γ::Vector{<:GenericSite}, Λ::Vector{<:LinkSite}; truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION, error = 0.0)
	UMPS{eltype(data(Γ[1]))}(Γ, Λ, truncation, error)
end

function UMPS(Γ::Vector{<:GenericSite}, Λ::Vector{<:LinkSite}, mps::UMPS; error=0)
	return UMPS{eltype(data(Γ[1]))}(Γ, Λ, mps.truncation,mps.error + error)
end

function UMPS(Γ::Vector{Array{T,3}}, mps::UMPS; error=0) where {T}
	Λ = LinkSite.([ones(T,size(γ,1))/sqrt(size(γ,1)) for γ in Γ])
	return UMPS(GenericSite.(Γ, ispurification(mps)), Λ, truncation= mps.truncation, error = mps.error + error)
end

function UMPS(sites::Vector{OrthogonalLinkSite{T}}; truncation, error = 0.0) where {T}
    Γ, Λ = ΓΛ(sites)
    UMPS(Γ,Λ[1:end-1], truncation=truncation, error = error)
end

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
function identityUMPS(N, d; T=ComplexF64, truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION)
	Γ = Vector{Array{T,3}}(undef,N)
	Λ = Vector{Vector{T}}(undef,N)
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
	if ispurification(mps)
        d = Int(sqrt(d))
    end
	trunc = mps.truncation
	return identityUMPS(N, d,T=T, truncation=trunc)
end

function productUMPS(theta, phi)
    Γ = [reshape([cos(theta), exp(phi*im)*sin(theta)],(1,2,1))]
    Λ = [[ComplexF64(1.0)]]
    return UMPS(Γ,Λ, purification = false)
end

Base.copy(mps::UMPS{T}) where {T} = UMPS{T}([copy(getfield(mps, k)) for k = 1:length(fieldnames(UMPS))]...) 

"""
	reverse_direction(mps::UMPS)

Flip the spatial direction
"""
function reverse_direction(mps::UMPS)
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
function transfer_matrix(mps::UMPS, gate::AbstractSquareGate, site::Integer, direction = :left)
	oplength = length(gate)
	if ispurification(mps)
		gate = auxillerate(gate)
	end
    return transfer_matrix(mps[site:site+oplength-1],op,direction)
end


"""
	transfer_spectrum(mps::UMPS, direction=:left; nev=1)

Return the spectrum of the transfer matrix of the UMPS
"""
function transfer_spectrum(mps::UMPS{K}, direction=:left; nev=1) where {K}
	# if K == ComplexDF64
	# 	@warn("converting ComplexDF64 to ComplexF64")
	# 	mps = convert(UMPS{ComplexF64},mps)
	# end
    T = transfer_matrix(mps,direction)
	D = Int(sqrt(size(T,2)))
	nev = minimum([D^2, nev])
    if D<4
        vals, vecs = eigen(Matrix(T))
        vals = vals[end:-1:1]
        vecs = vecs[:,end:-1:1]
    else
		x0 = vec(Matrix{K}(I,D,D))
        vals, vecsvec = eigsolve(T,x0,nev, :LM)#eigs(T,nev=nev)
		vecs = hcat(vecsvec...)
    end
	# if K == ComplexDF64
	# 	vals = ComplexDF64.(vals)
	# 	vecs = ComplexDF64.(vecs)
	# end
	# tensors = Array{Array{eltype(vecs),2},1}(undef,nev)
	# for i in 1:nev
	# 	tensors[i] = reshape(vecs[:,i],D,D)
	# end
	nev = min(length(vals),nev)
	tensors =  [reshape(vecs[:,k],D,D) for k in 1:nev]
    return vals[1:nev], tensors #canonicalize_eigenoperator.(tensors)
end

"""
	transfer_spectrum(mps::UMPS, mpo::AbstractMPO, direction=:left; nev=1)

Return the spectrum of the transfer matrix of the UMPS, with mpo sandwiched
"""
function transfer_spectrum(mps::UMPS{K}, mpo::AbstractMPO, direction=:left; nev=1) where {K}
    T = transfer_matrix(mps, mpo, direction)
	DdD = size(T,1)
	d = size(mpo[1],1)
	D = Int(sqrt(N/d))
	nev = minimum([DdD, nev])
    if N<10
        vals, vecs = eigen(Matrix(T))
        vals = vals[end:-1:end-nev+1]
        vecs = vecs[:,end:-1:end-nev+1]
    else
		x0id = Matrix{K}(I,D,D)
		x0v = rand(K,size(mpo[1],1))
		@tensor x0tens[:] := x0id[-1,-3]*x0v[-2]
		x0 = vec(x0tens)
        vals, vecsvec = eigsolve(T,x0,nev)#eigs(T,nev=nev)
		vecs = hcat(vecsvec...)
    end
	tensors =  [reshape(vecs[:,k],D,d,D) for k in 1:nev]
    return vals, tensors #canonicalize_eigenoperator.(tensors)
end

function LinearAlgebra.norm(mps::UMPS) #FIXME dont assume canonical
	return sum(data(mps.Λ[1]) .^2)
end


function apply_layers(mps::UMPS, layers)
    sites, err = apply_layers!(mps[1:end], layers, mps.truncation, isperiodic=true)
    return UMPS(sites, truncation=mps.truncation, error = mps.error + err)
end
function apply_layers_nonunitary(mps::UMPS, layers)
    sites, err = apply_layers_nonunitary(mps[1:end], layers, mps.truncation, isperiodic=true)
    return UMPS(sites, truncation=mps.truncation, error = mps.error + err)
end

# %% Canonicalize
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
	rhoR =  canonicalize_eigenoperator(rhoRs[1])
	rhoL =  canonicalize_eigenoperator(rhoLs[1])

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
	F = svd(Y*Diagonal(data(mps.Λ[1]))*transpose(X))

    #U,S,Vt,D,err = truncate_svd(F)

    #rest
    YU = VirtualSite(pinv(Y)*F.U ./ (valL[1])^(1/4))
    VX = VirtualSite(F.Vt*pinv(transpose(X)) ./ (valR[1])^(1/4))
	Γ[end] = Γ[end] * YU
	Γ[1] = VX*Γ[1]
    # @tensor Γ[end][:] := data(Γ[end])[-1,-2,3]*YU[3,-3]
    # @tensor Γ[1][:] := VX[-1,1]*data(Γ[1])[1,-2,-3]
	S = LinkSite(F.S)
	if mps.truncation.normalize
		Λ[1] = S / LinearAlgebra.norm(S)
	else
		Λ[1] = S
	end
	return
end

function canonicalize!(mps::UMPS)
	N = length(mps)
	if N>2
		error("Canonicalize with identity layers if the unit cell is larger than two sites")
	end
	canonicalize_cell!(mps)
	if N==2
		ΓL, ΓR, error = apply_two_site_gate(mps[1],mps[2], IdentityGate(2), mps.truncation)
	    #mps.Γ[1], mps.Λ[2], mps.Γ[2], err = apply_two_site_identity(mps.Γ, mps.Λ[mod1.(1:3,2)], mps.truncation)
		#mps.error += err
		Γ, Λ = ΓΛ([ΓL, ΓR]) 
		mps.Γ = Γ
		mps.Λ = Λ[1:end-1]
		mps.error += error
	end
    return mps
end

function canonicalize(mps::UMPS,n)
	for i in 1:n
		mps = apply_identity_layer(mps,0)
		mps = apply_identity_layer(mps,1)
	end
	return mps
end

"""
	boundary(mps::UMPS)

Return the left and right dominant eigentensors of the transfer matrix
"""
function boundary(mps::UMPS)
	valR, rhoRs = transfer_spectrum(mps,:left,nev=2)
	valL, rhoLs = transfer_spectrum(mps,:right,nev=2)
	DR = Int(sqrt(length(rhoRs[:,1])))
	DL = Int(sqrt(length(rhoLs[:,1])))
	rhoR =  reshape(rhoRs[:,1],DR,DR)
	rhoL =  reshape(rhoLs[:,1],DL,DL)
	return rhoL, rhoR
end
transfer_matrix_bond(mps::UMPS, site::Integer, dir::Symbol) = (s =Diagonal(data(mps.Λ[site])); kron(s,s))
transfer_matrix_bond(mps1::UMPS, mps2::UMPS, site::Integer, dir::Symbol) = kron(Diagonal(data(mps1.Λ[site])),Diagonal(data(mps2.Λ[site])))
transfer_matrix_bond(mps::ConjugateSite{<:UMPS}, site::Integer, dir::Symbol) = (s =Diagonal(data(mps.Λ[site])); kron(s,s))
transfer_matrix_bond(mps1::ConjugateSite{<:UMPS}, mps2::UMPS, site::Integer, dir::Symbol) = kron(Diagonal(data(mps1.Λ[site])),Diagonal(data(mps2.Λ[site])))

# """ 
# 	boundary(mps::UMPS, mpo::MPO) 

# Return the left and right dominant eigentensors of the transfer matrix
# """
# function boundary(mps::UMPS, mpo::AbstractMPO) #FIXME should implement https://arxiv.org/pdf/1207.0652.pdf
# 	valR, rhoRs = transfer_spectrum(mps,mpo,:left,nev=2)
# 	valL, rhoLs = transfer_spectrum(mps,mpo,:right,nev=2)
# 	DmpoR = size(mpo[end],4)
# 	DmpoL = size(mpo[1],1)
# 	DR = Int(sqrt(length(rhoRs[:,1])/DmpoR))
# 	DL = Int(sqrt(length(rhoLs[:,1])/DmpoL))
# 	rhoR =  reshape(rhoRs[:,1],DR,DmpoR,DR)
# 	rhoL =  reshape(rhoLs[:,1],DL,DmpoL,DL)
# 	return rhoL, rhoR
# end
function boundary(mps::UMPS,mpo::AbstractMPO, side::Symbol)
	_, rhos = transfer_spectrum(mps,mpo, reverse_direction(side),nev=2)
	return canonicalize_eigenoperator(rhos[1])
end

function boundary(mps::UMPS, side::Symbol)
	_, rhos = transfer_spectrum(mps, reverse_direction(side),nev=2)
	return canonicalize_eigenoperator(rhos[1])
end


#TODO Calculate expectation values and effective hamiltonian as in https://arxiv.org/pdf/1207.0652.pdf
#FIXME should implement https://arxiv.org/pdf/1207.0652.pdf
""" 
	effective_hamiltonian(mps::UMPS, mpo::MPO) 

Return the left and right effective_hamiltonian
"""
function effective_hamiltonian(mps::UMPS{T}, mpo::AbstractMPO; direction=:left) where {T}
	Dmpo = size(mpo[end],1)
	D = length(mps.Λ[1])
	sR = (D,Dmpo,D)
	TL = transfer_matrix(mps,mpo,direction)
	TIL = transfer_matrix(mps,direction)
	@warn "Make sure that mpo is lower triangular with identity on the first and last place of the diagonal"
	rhoR = zeros(T,sR) #TODO Sparse array?
	itr = 1:Dmpo
	if direction ==:right 
		itr = reverse(itr)
	end
	rhoR[:,itr[end],:] = Matrix{T}(I,D,D)
	for k in Dmpo-1:-1:1
		rhoR[:,itr[k],:] = reshape(TL * vec(rhoR),sR)[:,itr[k],:]
	end
	C = rhoR[:,itr[1],:]
	rho = Diagonal(data(mps.Λ[1]).^2)
	@tensor e0[:] := C[1,2]*rho[1,2]
	idvec = vec(Matrix{T}(I,D,D))
	function TI(v)
		v = TIL*v
		return (v - idvec *(vec(rho)'*v))
	end
	linmap = LinearMap{ComplexF64}(TI,D^2)
	hl, info = linsolve(linmap,vec(C)-e0[1]*idvec,1,-1)
	rhoR[:,itr[1],:] = hl
	return e0[1], rhoR ,info
end



"""
	canonicalize_cell(mps::UMPS)

Make the unit cell canonical and return the resulting UMPS
"""
function canonicalize_cell(mps::UMPS)
	D = length(mps.Λ[1])
	N = length(mps.Γ)
	Γcopy = copy(mps.Γ)
	Λcopy = copy(mps.Λ)

	valR, rhoRs = transfer_spectrum(mps,:left,nev=2)
	valL, rhoLs = transfer_spectrum(mps,:right,nev=2)
	rhoR =  canonicalize_eigenoperator(rhoRs[1])
	rhoL =  canonicalize_eigenoperator(rhoLs[1])

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
	F = svd!(Y*Diagonal(data(mps.Λ[1]))*transpose(X))

    U,S,Vt,D,err = truncate_svd(F, mps.truncation)
	Λ = LinkSite(S)
    #rest
    YU = VirtualSite(pinv(Y)*U ./ (valL[1])^(1/4))
    VX = VirtualSite(Vt*pinv(transpose(X)) ./ (valR[1])^(1/4))
	Γcopy[end] = mps.Γ[end]*YU
	Γcopy[1] = VX*mps.Γ[1]
    # @tensor Γcopy[end][:] := Γcopy[end][-1,-2,3]*YU[3,-3]
    # @tensor Γcopy[1][:] := VX[-1,1]*Γcopy[1][1,-2,-3]
	if mps.truncation.normalize
		Λcopy[1] = Λ / LinearAlgebra.norm(Λ)
	else
		Λcopy[1] = Λ
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
	elseif N==2
	    #Γ[1],Λ2,Γ[2], err = apply_two_site_identity(mps.Γ, mps.Λ[mod1.(1:3,2)], mps.truncation)
		ΓL, ΓR, err = apply_two_site_gate(mps[1],mps[2], IdentityGate(2), mps.truncation)
		mpsout = UMPS([ΓL, ΓR], truncation=mps.truncation, error = err)
	else
		error("Canonicalizing $N unit sites not implemented")
		return mps
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



#%% Expectation values
function expectation_value(mps::UMPS, op::Array{T_op,N_op}, site::Integer) where {T_op<:Number,N_op}
	opLength=length(op)
	N = length(mps.Γ)
	if ispurification(mps)
		op = auxillerate(op)
	end
	if opLength == 1
		val = expectation_value_one_site(mps.Λ[site],mps.Γ[site],mps.Λ[mod1(site+1,N)],op)
	elseif opLength == 2
		val = expectation_value_two_site(mps.Γ[mod1.(site:site+1,N)],mps.Λ[mod1.(site:site+2,N)],op)
	else
		error("Expectation value not implemented for operators of this size")
	end
	return val
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
	rhoH = Hermitian((rho + rho')/2)
    return rhoH/tr(rhoH) * size(rhoH,1)
end


# %% Entropy
# function renyi(mps::UMPS, n)
# 	N = length(mps.Γ)
# 	T = eltype(mps.Γ[1])
#     transfer_matrices = transfer_matrices_squared(mps,:right)
# 	sizes = size.(mps.Γ)
# 	id = Matrix{T}(I,sizes[1][1],sizes[1][1])
# 	leftVec = vec(@tensor id[-1,-2]*id[-3,-4])
# 	Λsquares = Diagonal.(mps.Λ .^2)
# 	rightVecs = map(k->vec(@tensor Λsquares[-1,-2]*Λsquares[-3,-4])', 1:N)
#     vals = Float64[]
#     for k in 1:n
# 		leftVec = transfer_matrices[mod1(k,N)]*leftVec
# 		val = rightVecs[mod1(k+1,N)] * leftVec
#         push!(vals, -log2(val))
#     end
#     return vals
# end

# function renyi(mps::UMPS) #FIXME implement transfer_matrix_squared
# 	N = length(mps.Γ)
# 	T = eltype(mps.Γ[1])
#     transfer_matrix = transfer_matrix_squared(mps,:right)
# 	return -log2(eigsolve(transfer_matrix,1)[1])
# end

# %%
function saveUMPS(mps, filename)
    jldopen(filename, "w") do file
		writeOpenMPS(file,mps)
    end
end

function writeUMPS(parent, mps)
	write(parent, "Gamma", mps.Γ)
	write(parent, "Lambda", mps.Λ)
	write(parent, "Purification", ispurification(mps))
	write(parent, "Dmax", mps.truncation.Dmax)
	write(parent, "tol", mps.truncation.tol)
	write(parent, "normalize", mps.truncation.normalize)
	write(parent, "error", mps.error)
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
