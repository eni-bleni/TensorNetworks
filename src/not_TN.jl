
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
