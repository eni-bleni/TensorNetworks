
"""
	LRcanonical(M, dir)-> A,R,DB

Return the left or right canonical form of a single tensor.

:left is leftcanonical, :right is rightcanonical
"""
function LRcanonical(M, dir=:left)
    if dir == :right
        M = permutedims(M,[3,2,1])
    end
    D1,d,D2 = size(M)
    M = reshape(M,D1*d,D2)
    A,R = qr(M) # M = Q R
	A = Matrix(A)
	R = Matrix(R)
    Db = size(R,1) # intermediate bond dimension
	A = reshape(A,D1,d,Db)
    if dir == :right
        A = permutedims(A,[3,2,1])
        R = transpose(R)
    end
    return A,R,Db
end

"""
	truncate_svd(F, args)

Truncate an SVD object
"""
function truncate_svd(F, args::TruncationArgs)
	Dmax = args.Dmax
	tol = args.tol
	D = min(Dmax,length(F.S))
	S = F.S[1:D]
	err = sum(F.S[D+1:end].^2) + sum(S[S .< tol].^2)
	S = S[S .> tol]
	if args.normalize
		S = S ./ LinearAlgebra.norm(S)
	end
	D = length(S)
	@views return F.U[:, 1:D], S, F.Vt[1:D, :], D,err
end

"""
	check_LRcanonical(tensor,dir)

check whether given site/tensor in mps is left-/rightcanonical
"""
function check_LRcanonical(a::GenericSite, dir)
    if dir == :left
        @tensor c[-1,-2] := a.Γ[1,2,-2]*conj(a.Γ[1,2,-1])
    elseif dir == :right
        @tensor c[-1,-2] := a.Γ[-1,2,1]*conj(a.Γ[-2,2,1])
    else
        println("ERROR: choose :left for leftcanonical or :right for rightcanonical")
        return false
    end
    return c ≈ Matrix(1.0I,size(c,1),size(c,1))
end

"""
	split_truncate(tensor, args)

Split and truncate a two-site tensor
"""
function split_truncate(theta, args::TruncationArgs)
	D1l,d,d,D2r = size(theta)
    theta = reshape(theta, D1l*d,d*D2r)
	F = try
        svd!(theta)
    catch y
        svd!(theta,alg=LinearAlgebra.QRIteration())
    end
    U,S,Vt,Dm,err = truncate_svd(F, args)
    return U,S,Vt,Dm,real(err)
end


"""
	apply_two_site_gate(Γ, Λ, gate, Dmax,tol)

Act with a two site gate and return the truncated decomposition U,S,Vt,err
"""
function apply_two_site_gate(Γ, Λ, gate::GenericSquareGate, args::TruncationArgs)
	ΓL = similar(Γ[1])
	ΓR = similar(Γ[2])
	absorb_l!(ΓL, Λ[1], Γ[1], Λ[2])
	absorb_l!(ΓR,Γ[2], Λ[3])
    @tensoropt (5,-1,-4) theta[:] := ΓL[-1,2,5]*ΓR[5,3,-4]*gate.data[-2,-3,2,3]
	DL,d,d,DR = size(theta)

	U,S,Vt,Dm,err = split_truncate(theta, args)

	U=reshape(U,DL,d,Dm)
	Vt=reshape(Vt,Dm,d,DR)
    U = absorb_l(U, 1 ./Λ[1],:left)
    Vt = absorb_l(Vt,1 ./Λ[3],:right)
    return U,S,Vt,err
end

"""
	apply_two_site_identity(Γ, Λ, Dmax,tol)

Act with a two site identity gate and return the truncated decomposition U,S,Vt,err
"""
function apply_two_site_identity(Γ, Λ, args::TruncationArgs)
	ΓL = similar(Γ[1])
	ΓR = similar(Γ[2])
	absorb_l!(ΓL, Λ[1], Γ[1], Λ[2])
	absorb_l!(ΓR, Γ[2], Λ[3])

    @tensor theta[:] := ΓL[-1,-2,5]*ΓR[5,-3,-4]
	DL,d,d,DR = size(theta)

	U,S,Vt,Dm,err = split_truncate(theta,args)
	U=reshape(U,DL,d,Dm)
	Vt=reshape(Vt,Dm,d,DR)
    U = absorb_l(U, 1 ./Λ[1],:left)
    Vt = absorb_l(Vt,1 ./Λ[3],:right)

    return U,S,Vt,err
end
"""
	absorb_l!

Absorb the link tensor into the main tensor
"""
function absorb_l!(gout, l::Array{T,1}, g) where {T}
    sg=size(g)
	gout .= reshape(l,sg[1],1,1) .* g
end
function absorb_l!(gout,g , l::Array{T,1}) where {T}
    sg=size(g)
	gout .= g .* reshape(l,1,1,sg[3])
end
function absorb_l!(gout,ll,g,lr)
    sg=size(g)
	gout .= reshape(ll,sg[1],1,1) .* g .*reshape(lr,1,1,sg[3])
end

function absorb_l(ll::Vector{T},g,lr::Vector{T}) where {T}
    sg=size(g)
	return reshape(ll,sg[1],1,1) .* g .*reshape(lr,1,1,sg[3])
end
function absorb_l(g,l,dir::Symbol=:left)
    sg=size(g)
    if dir == :left
		s = reshape(l,sg[1],1,1)
    elseif dir==:right
		s = reshape(l,1,1,sg[3])
    end
    return g .* s
end

"""
	deauxillerate_onesite

Reshape a purification to a density matrix
"""
function deauxillerate_onesite(tens)
    s = size(tens)
    d=Int(sqrt(s[2]))
    return reshape(tens,s[1],d,d,s[3])
end

# """
# 	block_to_gate

# Return the 4 legged version of the matrix
# """
# function block_to_gate(block)
#     d = Int(sqrt(size(block,1)))
#     return reshape(block,d,d,d,d)
# end

# function matrix_to_gate(matrix, n)
#     d = Int((size(matrix,1))^(1/n))
#     return reshape(matrix, repeat([d],n)...)
# end
# function gate_to_matrix(gate, n)
# 	dims = size(gate)
# 	op_length = operator_length(gate)
#     return reshape(gate, *(dims[1:op_length]...), *(dims[op_length+1:end]...))
# end


# %% Expectation values
function expectation_value_two_site(Γ,Λ,op)
    thetaL = absorb_l(Γ[1],Λ[1],:left)
    thetaL = absorb_l(thetaL,Λ[2],:right)
    thetaR = absorb_l(Γ[2],Λ[3],:right)
    @tensoropt (d,u,r,l) r[:] := thetaL[l,cld,d] *op[clu,cru,cld,crd] *conj(thetaL[l,clu,u]) *conj(thetaR[u,cru,r]) *thetaR[d,crd,r]
    return r[1]
end

function expectation_value_two_site(Γ,op)
    @tensoropt (d,u,r,l) r[:] := thetaL[l,cld,d] *op[clu,cru,cld,crd] *conj(thetaL[l,clu,u]) *conj(thetaR[u,cru,r]) *thetaR[d,crd,r]
    return r[1]
end

"""
	local_ham_eigs(hamiltonian, size, nev=2)

Calculate the 'nev' lowest eigenstates

See also: [`local_ham_eigs_threads`](@ref)
"""
function local_ham_eigs(ham_gate::AbstractArray{T}, N; nev=2) where {T}
	Nd = Int(length(size(ham_gate))/2)
	d = size(ham_gate, 1)
	block = reshape(ham_gate, d^Nd, d^Nd)
	indices = 1:N
	tensify(x) = reshape(x,repeat([d],N)...)
	blockify(x) = reshape(x, d^Nd, d^(N-Nd))
	function apply_ham(invec)
		tens0 = tensify(invec)
		tens = zeros(T, size(tens0))
		for k in 1:N
			tens += permutedims(tensify( block * blockify( permutedims( tens0, circshift(indices,k)))), circshift(indices,-k))
		end
		return vec(tens)
	end
	map = LinearMap{T}(apply_ham, d^N ,ishermitian = true)
	# println(sparse(block))
	# println(sparse(map))
	# println(sparse(map)' - sparse(map))
	# shift(n) = translation_matrix(d,N,n)
	# println(translation_matrix(d,3,0))
	# println((translation_matrix(d,N,0)*sparse(Matrix(map))*translation_matrix(d,N,0)) - sparse(Matrix(map)))
	# println(tr(translation_matrix(d,N,0)*Matrix(map)-sparse(Matrix(map))))
	# mat = (d^N < 10 ? Matrix(map) : map)
	return eigsolve(map,d^N,nev,:SR,ishermitian=true) #eigs(map, nev=nev, which=:SR)
end

function local_ham_eigs_sparse(ham_mat::AbstractArray{T}, d, N; nev=2) where {T}
	Nh = Int(log(d,size(ham_mat,1)))
	id = sparse(1.0I,d^(N-Nh),d^(N-Nh))
	ham = kron(sparse(ham_mat),id)
	shift(n) = translation_matrix(d,N,n)
	shifts = shift.(1:N)
	function apply_ham(invec)
		sum = zeros(T, size(invec))
		for k in 1:N
			sum += shifts[N+1-k]*(ham*invec)
			invec = shifts[1]*invec
		end
		return sum
	end
	#println(shifts[N+1]*ham*shifts[1])
	map = LinearMap{T}(apply_ham, d^N,ishermitian = true)
	# mat = (d^N < 10 ? Matrix(map) : map)
	# println(sparse(map))
	# println(sparse(map)' - sparse(map))
	# println(shift(-1)*sparse(map)*shift(1) - sparse(map))
	return eigsolve(map,d^N,nev,:SR,ishermitian=true) #eigs(map, nev=nev, which=:SR)
end
using Combinatorics 
function translation_tensor(d,N,n=1) #FIXME d=2,N=4,n=0 has a bug!
	sa = SparseArray{Float64}(undef, tuple(repeat([d],2*N)...))
	for combination in Combinatorics.with_replacement_combinations(1:d,N)
		for perm in Combinatorics.multiset_permutations(combination,N)
			shifted = perm
			# while true
				sa[tuple(circshift(shifted,n)..., shifted...)...] = 1.0
				shifted = circshift(shifted,1)
				# (shifted != combination) || break
			# end
		end
	end
	return sa
end
#translation_matrix2(d,N) = reshape(translation_tensor(d,N),d^N,d^N)
function translation_matrix(d,N,n=1) #FIXME d=2,N=4,n=0 has a bug!
	sa = SparseArray{Float64}(undef, tuple(repeat([d],N)...))
	sm = spzeros(d^N,d^N)
	linearIndices = LinearIndices(sa)
	for combination in Combinatorics.with_replacement_combinations(1:d,N)
		for perm in Combinatorics.multiset_permutations(combination,N)
			shifted = perm
			# while true
				sm[linearIndices[circshift(shifted,n)...], linearIndices[shifted...]] = 1.0
				shifted = circshift(shifted,1)
				# (shifted != combination) || break
			# end
		end
	end
	return sm
end


function local_ham_eigs2(ham_mat, d, N; nev=2)
	Nh = Int(log(2,size(ham_mat,1)))
	id = sparse(1.0I,d^(N-Nh),2^(N-Nh))
	ham = kron(ham_mat,id)
	bigham(n1) = kron(sparse(1.0I,d^(N-Nh-n1),2^(N-Nh-n1)),ham_mat,sparse(1.0I,d^(n1),2^(n1)))
	bighams = [bigham(n) for n in 0:N-2]
	shift = kron(sparse(1.0I,d,d), circshift(sparse(1.0I,d^(N-1),d^(N-1)),1))
	function apply_ham(invec)
		sum = zeros(eltype(ham_mat), size(invec))
		for k in 1:N-1
			sum += bighams[k]*invec
			#invec = shift*invec
		end
		return sum
	end
	map = LinearMap{eltype(ham_mat)}(apply_ham, d^N, ishermitian = true)
	mat = (d^N < 10 ? Matrix(map) : map)
	return eigsolve(map,d^N,nev,ishermitian=true) #eigs(map, nev=nev, which=:SR)
end


"""
	local_ham_eigs_threads(hamiltonian, size, nev=2)

Calculate the 'nev' lowest eigenstates

See also: [`local_ham_eigs`](@ref)
"""
function local_ham_eigs_threads(ham_gate, N; nev=2)
	Nd = Int(length(size(ham_gate))/2)
	d = size(ham_gate, 1)
	block = reshape(ham_gate, d^Nd, d^Nd)
	indices = 1:N
	tensify(x) = reshape(x,repeat([d],N)...)
	blockify(x) = reshape(x, d^Nd, d^(N-Nd))
	function apply_ham(invec)
		tens0 = tensify(invec)
		vecs = Array{Vector{eltype(ham_gate)},1}(undef,N)
		Threads.@threads for k in 1:N
			vecs[k] = vec(permutedims(tensify( block * blockify( permutedims( tens0, circshift(indices,k)))), circshift(indices,-k)))
		end
		return sum(vecs)
	end

	map = LinearMap{eltype(ham_gate)}(apply_ham, d^N, ishermitian = true)
	mat = (d^N < 10 ? Matrix(map) : map)
	return eigsolve(map,d^N,nev,ishermitian=true)#eigs(map, nev=nev)
end
