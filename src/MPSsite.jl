Base.permutedims(site::GenericSite, perm) = GenericSite(permutedims(site.Γ,perm), site.purification)
Base.copy(site::OrthogonalLinkSite) = OrthogonalLinkSite([copy(getfield(site, k)) for k = 1:length(fieldnames(OrthogonalLinkSite))]...) 
Base.copy(site::GenericSite) = GenericSite([copy(getfield(site, k)) for k = 1:length(fieldnames(GenericSite))]...) 

Base.@propagate_inbounds Base.getindex(site::AbstractSite, I...) = getindex(data(site),I...)
Base.@propagate_inbounds Base.getindex(site::AdjointSite, I...) = getindex(site.parent,I...)

Base.size(site::AbstractSite, dim) = size(data(site), dim)
Base.size(site::AbstractSite) = size(data(site))
Base.size(site::AdjointSite) = size(site.parent)
Base.size(site::AdjointSite, dim) = size(site.parent,dim)
Base.size(site::OrthogonalLinkSite) = size(site.Γ)
Base.size(site::OrthogonalLinkSite, dim) = size(site.Γ, dim)
Base.size(site::VirtualSite, dim) = size(site.Λ, dim)
Base.size(site::VirtualSite) = size(site.Λ)
Base.length(site::LinkSite) = length(site.Λ)

#Base.isapprox(s1::AbstractSite,s2::AbstractSite) = isapprox(data(s1),data(s2))
Base.isapprox(s1::OrthogonalLinkSite, s2::OrthogonalLinkSite) = isapprox(s1.Γ, s2.Γ) && isapprox(s1.Λ1, s2.Λ1) && isapprox(s1.Λ2, s2.Λ2)
ispurification(site::GenericSite) = site.purification
ispurification(site::OrthogonalLinkSite) = ispurification(site.Γ)

link(site::GenericSite, dir) = I
link(site::AdjointSite, dir) = link(site.parent, dir)'
function link(site::OrthogonalLinkSite,dir)
	if dir==:left
		site.Λ1
	elseif dir == :right
		site.Λ2
	else 
		error("Choose direction :left or :right")
	end
end
data(site::GenericSite) = site.Γ
data(site::VirtualSite) = site.Λ
data(site::LinkSite) = site.Λ
data(site::AdjointSite) = conj(data(site.parent))
data(site::GenericSite,dir) = site.Γ
data(site::OrthogonalLinkSite, dir) = data(GenericSite(site,dir))
data(site::VirtualSite, dir) = site.Λ
data(site::LinkSite, dir) = site.Λ
data(site::AdjointSite,dir) = conj(data(site.parent,dir))
data(site::OrthogonalLinkSite) = data(site.Λ1*site.Γ*site.Λ2)
# data(site::AdjointSite) = conj(data(site.site))

MPOsite(site::GenericSite) = (s = size(site); MPOsite(reshape(data(site),s[1],s[2],1,s[3])))
MPOsite(site::GenericSite, dir) = MPOsite(site)
MPOsite(site::AdjointSite) = conj(permutedims(MPOsite(site.parent),[1,3,2,4]))
MPOsite(site::AdjointSite, dir) =  conj(permutedims(MPOsite(site.parent,dir),[1,3,2,4]))
MPOsite(site::OrthogonalLinkSite,dir) = MPOsite(GenericSite(site,dir),dir)

MPOsite{K}(site::GenericSite) where {K} = (s = size(site); MPOsite{K}(reshape(data(site),s[1],s[2],1,s[3])))
MPOsite{K}(site::GenericSite, dir) where {K} = MPOsite{K}(site)
MPOsite{K}(site::AdjointSite) where {K} = conj(permutedims(MPOsite{K}(site.parent),[1,3,2,4]))
MPOsite{K}(site::AdjointSite, dir) where {K} =  conj(permutedims(MPOsite{K}(site.parent,dir),[1,3,2,4]))
MPOsite{K}(site::OrthogonalLinkSite,dir) where {K} = MPOsite{K}(GenericSite(site,dir),dir)
# MPOsite(site::BraSite{<:OrthogonalLinkSite},dir) = MPOsite(GenericSite(site.site,dir)',dir)
#MPOsite(site::AbstractSite, dir::Symbol) = MPOsite(GenericSite(site,dir))

# Base.adjoint(site::AbstractSite) = BraSite(site)
# Base.adjoint(site::BraSite) = site.site

Base.sqrt(site::LinkSite) = LinkSite(sqrt(data(site)))

ispurification(site::AdjointSite) = ispurification(site.parent)

isleftcanonical(site::AbstractSite)  = isleftcanonical(data(site))
isleftcanonical(site::OrthogonalLinkSite) = isleftcanonical(site.Λ1*site.Γ)
isrightcanonical(site::AbstractSite) = isrightcanonical(data(site))
isrightcanonical(site::OrthogonalLinkSite) = isrightcanonical(site.Γ*site.Λ2)
iscanonical(site::OrthogonalLinkSite) = isrightcanonical(site) && isleftcanonical(site) && norm(site.Λ1) ≈ 1 && norm(site.Λ2) ≈ 1

entanglement_entropy(Λ::LinkSite) = -sum(data(Λ) * log(data(Λ)))

GenericSite(site::GenericSite) = site
GenericSite(site::GenericSite,dir) = site
GenericSite(site::AdjointSite) = GenericSite(conj(data(site.parent)), ispurification(site))

Base.convert(::Type{LinkSite{T}}, Λ::LinkSite{K}) where {K,T} = LinkSite(convert.(T,Λ.Λ))
Base.convert(::Type{GenericSite{T}}, site::GenericSite{K}) where {K,T} = GenericSite(convert.(T,data(site)),ispurification(site))


LinearAlgebra.norm(site::AbstractSite) = norm(data(site))
# LinearAlgebra.norm(site::LinkSite) = norm(data(site))
# LinearAlgebra.norm(site::VirtualSite) = norm(data(site))

# Base.eltype(site::BraSite) = eltype(site.site)
Base.eltype(::AbstractSite{T,N}) where {T,N} = T
Base.eltype(::AdjointSite{T,N,S}) where {T,N,S} = T
# Base.eltype(::LinkSite{T}) where {T} = T
# Base.eltype(::OrthogonalLinkSite{T}) where {T} = T
# Base.eltype(::VirtualSite{T}) where {T} = T

function LinearAlgebra.ishermitian(site::MPOsite)
	ss = size(site)
	if !(ss[1] == 1 && ss[4] == 1)
		return false
	else
		m =reshape(data(site),ss[2],ss[3])
		return ishermitian(m)
	end
end

function LeftOrthogonalSite(site::OrthogonalLinkSite; check=true) 
	L = site.Λ1*site.Γ
	check || @assert isleftcanonical(L) "In LeftOrthogonalSite: site is not left canonical"
	return L
end
function RightOrthogonalSite(site::OrthogonalLinkSite; check=true) 
	R = site.Γ*site.Λ2
	check || @assert isrightcanonical(R) "In RightOrthogonalSite: site is not right canonical"
	return R
end
function OrthogonalSite(site::OrthogonalLinkSite, side; check=true) 
	if side==:left
		return LeftOrthogonalSite(site,check=check)
	elseif side==:right
		return RightOrthogonalSite(site,check=check)
	else 
		"Error in OrthogonalSite: choose side :left or :right"
	end
end

"""
to_left_orthogonal(site::GenericSite, dir)-> A,R,DB

Return the left orthogonal form of the input site as well as the remainder.
"""
function to_left_orthogonal(site::GenericSite; full=false, method=:qr, truncation=DEFAULT_OPEN_TRUNCATION)
    D1,d,D2 = size(site)
    M = reshape(data(site),D1*d,D2)
	if method ==:qr
    	U,R = qr(M) 
		A = full ? U * Matrix(I,D2,D2) : Matrix(U)
	elseif method==:svd
		U, S, Vt, Dm, err = split_truncate!(copy(M), truncation)
		A = Matrix(U)
		R = Diagonal(S)*Vt
	else
		error("Choose :qr or :svd as method in 'to_left_orthogonal'")
	end
    Db = size(R,1) # intermediate bond dimension
	orthSite = GenericSite(reshape(A,D1,d,Db), site.purification)
	V = VirtualSite(R)
    return orthSite, V
end
function to_right_orthogonal(site::GenericSite; full=false, method=:qr)
	#M = permutedims(site,[3,2,1])
	L, V = to_left_orthogonal(reverse_direction(site), full=full,method=method)
	reverse_direction(L), transpose(V)
end

function LinearAlgebra.svd(site::GenericSite, orth = :leftorthogonal)
	s = size(site)
	if orth == :leftorthogonal
		m = reshape(data(site),s[1]*s[2],s[3])
		F = svd(m)
		U = GenericSite(reshape(F.U,s),site.purification)
		S = LinkSite(F.S)
		Vt = VirtualSite(F.Vt)
	elseif orth == :rightorthogonal
		m = reshape(data(site),s[1],s[2]*s[3])
		F = svd(m)
		U = VirtualSite(F.U)
		S = LinkSite(F.S)
		Vt = GenericSite(reshape(F.Vt,s),site.purification)
	else
		error("Choose :leftorthogonal or :rightorthogonal")
	end
	return U,S,Vt
end

reverse_direction(site::GenericSite) = GenericSite(permutedims(data(site),[3,2,1]), site.purification)
reverse_direction(site::AdjointSite) = reverse_direction(site.parent)'

Base.transpose(G::VirtualSite) = VirtualSite(Matrix(transpose(G.Λ)))
Base.transpose(Λ::LinkSite) = Λ

Base.:*(Γ::GenericSite, α::Number) = GenericSite(α*data(Γ),Γ.purification)
Base.:*(α::Number, Γ::GenericSite) = GenericSite(α*data(Γ),Γ.purification)
Base.:/(Γ::GenericSite, α::Number) = GenericSite(data(Γ)/α,Γ.purification)
Base.:*(Λ::LinkSite, G::VirtualSite) = VirtualSite(reshape(diag(Λ.Λ),size(G,1),1) .* G.Λ)
Base.:*(G::VirtualSite, Λ::LinkSite) = VirtualSite(reshape(diag(Λ.Λ),1,size(G,2)) .* G.Λ)
Base.:/(Γ::LinkSite, α::Number) = LinkSite(data(Γ)/α)
Base.:/(Γ::VirtualSite, α::Number) = VirtualSite(data(Γ)/α)


Base.:*(Λ::LinkSite, Γ::GenericSite) = GenericSite(reshape(diag(Λ.Λ),size(Γ,1),1,1) .* data(Γ), Γ.purification)
Base.:*(Γ::GenericSite, Λ::LinkSite) = GenericSite(data(Γ) .* reshape(diag(Λ.Λ),1,1,size(Γ,3)), Γ.purification)
function Base.:*(G::VirtualSite, Γ::GenericSite)
	sG = size(G)
	sΓ = size(Γ)
	Γnew = reshape(data(G) * reshape(data(Γ),sΓ[1],sΓ[2]*sΓ[3]), sG[1], sΓ[2], sΓ[3])
	# @tensor Γnew[:] := data(G)[-1,1] * data(Γ)[1,-2,-3]
	GenericSite(Γnew, Γ.purification)
end
function Base.:*(Γ::GenericSite, G::VirtualSite)
	sG = size(G)
	sΓ = size(Γ)
	Γnew = reshape(reshape(data(Γ),sΓ[1]*sΓ[2],sΓ[3]) * data(G), sΓ[1], sΓ[2], sG[2])
	# @tensor Γnew[:] := data(Γ)[-1,-2,1] * data(G)[1,-3]
	GenericSite(Γnew, Γ.purification)
end

Base.inv(G::VirtualSite) = VirtualSite(inv(G.Λ))
Base.inv(Λ::LinkSite) = LinkSite(inv(Λ.Λ))

function Base.:*(gate::GenericSquareGate, Γ::Tuple{GenericSite, GenericSite})
	g = data(gate)
	s1,s2 = size.(Γ)
	m1 = reshape(data(Γ[1]),s1[1]*s1[2],s1[3])	
	m2 = reshape(data(Γ[2]),s2[1],s2[2]*s2[3])
	m12 = reshape(m1*m2,s1[1],s1[2],s2[2],s2[3])
	return @tullio theta[l,lu,ru,r] := m12[l,cl,cr,r]*g[lu,ru,cl,cr]
	#m12 = reshape(permutedims(reshape(m1*m2,s1[1],s1[2],s2[2],s2[3]),[2,3,1,4]), s1[2]*s2[2],s1[1]*s2[3])
	#return permutedims(reshape(reshape(g,sg[1]*sg[2],sg[3]*sg[4]) *m12,s1[2],s2[2],s1[1],s2[3]), [3,1,2,4])
	#@tensoropt (5,-1,-4) theta[:] := L[-1,2,5]*R[5,3,-4]*g[-2,-3,2,3]
end
function Base.:*(gate::ScaledIdentityGate{<:Number,4}, Γ::Tuple{GenericSite, GenericSite})
	s1,s2 = size.(Γ)
	m1 = reshape(data(Γ[1]),s1[1]*s1[2],s1[3])	
	m2 = reshape(data(Γ[2]),s2[1],s2[2]*s2[3])	
	return reshape(rmul!(m1*m2,data(gate)),s1[1],s1[2],s2[2],s2[3]) # Fast at runtime and at compilation
	#return data(gate)*m1*m2
	#return reshape(data(gate)*m1*m2,s1[1],s1[2],s2[2],s2[3])
	#@tensor theta[:] := data(gate)*data(Γ[1])[-1,-2,1]*data(Γ[2])[1,-3,-4]
end	

# function Base.:*(gate::AbstractSquareGate, Γ::Tuple{OrthogonalLinkSite, OrthogonalLinkSite})
# 	@assert Γ[1].Λ2 == Γ[2].Λ1 "Error in applying two site gate: The sites do not share a link"
# 	ΓL = LeftOrthogonalSite(Γ1)
# 	ΓR = Γ2.Λ1*RightOrthogonalSite(Γ2)
# 	gate*(ΓL,ΓR)
# end

OrthogonalLinkSite(Γ::GenericSite, Λ1::LinkSite, Λ2::LinkSite; check=false) = OrthogonalLinkSite(Λ1, Γ, Λ2, check=check)

"""
	compress(Γ1::GenericSite, Γ2::GenericSite, args::TruncationArgs)

Contract and compress the two sites using the svd. Return two U,S,V,err where U is a LeftOrthogonalSite, S is a LinkSite and V is a RightOrthogonalSite
"""
compress(Γ1::AbstractSite, Γ2::AbstractSite,args::TruncationArgs) = apply_two_site_gate(Γ1,Γ2,IdentityGate(2), args)

"""
	apply_two_site_gate(Γ1::GenericSite, Γ2::GenericSite, gate, args::TruncationArgs)

Contract and compress the two sites using the svd. Return two U,S,V,err where U is a LeftOrthogonalSite, S is a LinkSite and V is a RightOrthogonalSite
"""
function apply_two_site_gate(Γ1::GenericSite, Γ2::GenericSite, gate, args::TruncationArgs)
	theta = gate*(Γ1,Γ2)
	DL,d,d,DR = size(theta)
	U,S,Vt,Dm,err = split_truncate!(reshape(theta,DL*d,d*DR),args)
	U2 = GenericSite(Array(reshape(U,DL,d,Dm)), ispurification(Γ1))
	Vt2 = GenericSite(Array(reshape(Vt,Dm,d,DR)), ispurification(Γ2))
	S2 = LinkSite(S)
    return U2, S2, Vt2, err
end

function apply_two_site_gate(Γ1::OrthogonalLinkSite, Γ2::OrthogonalLinkSite, gate, args::TruncationArgs)
	@assert Γ1.Λ2 ≈ Γ2.Λ1 "Error in apply_two_site_gate: The sites do not share a link"
	ΓL = LeftOrthogonalSite(Γ1)
	ΓR = Γ2.Λ1*RightOrthogonalSite(Γ2)
	U,S,Vt,err = apply_two_site_gate(ΓL, ΓR, gate, args)
	U2 = inv(Γ1.Λ1)*U
    Vt2 = Vt*inv(Γ2.Λ2)
	Γ1new = OrthogonalLinkSite(Γ1.Λ1, U2, S)
	Γ2new = OrthogonalLinkSite(S, Vt2, Γ2.Λ2)
    return Γ1new, Γ2new, err
end

function to_left_right_orthogonal(M::Vector{GenericSite{T}}; center=1, method=:qr) where {T}
	N = length(M)
    @assert N+1>=center>=0 "Error in 'to_left_right_orthogonal': Center is not within the chain, center==$center"
	M = deepcopy(M)
	Γ = similar(M)
    local G::VirtualSite{T}
    for i in 1:center-1
        Γ[i], G = to_left_orthogonal(M[i], method= method)
        i<N && (M[i+1] = G*M[i+1])
    end
    for i in N:-1:center+1
        Γ[i], G = to_right_orthogonal(M[i], method = method)
        i>1 && (M[i-1] = M[i-1]*G)
    end
	
	if center == 0 || center==N+1
		if !(data(G) ≈ ones(T,1,1)) 
			@warn "In to_orthogonal!: remainder is not 1 at the end of the chain. $G"
		end
	else
		Γ[center] = M[center]/norm(M[center])
	end
    return Γ
end
function to_right_orthogonal(M::Vector{GenericSite{T}}; method=:qr) where {T}
    out = Vector{GenericSite{T}}(undef,length(M))
	M2 = copy(M)
	N = length(M)
    local G::VirtualSite{T}
    for i in N:-1:1
        out[i], G = to_right_orthogonal(M2[i], method=method)
 		if i>1
			M2[i-1] = M2[i-1]*G
		end
    end
    return out, G
end

function GenericSite(site::OrthogonalLinkSite, direction = :left)
	if direction==:left
		return site.Λ1*site.Γ
	elseif direction==:right
		return site.Γ*site.Λ2
	end
end

function ΓΛ(sites::Vector{OrthogonalLinkSite{T}}) where {T}
	N = length(sites)
	Γ = Vector{GenericSite{T}}(undef,N)
    Λ = Vector{LinkSite{T}}(undef,N+1)
    for k in 1:N
        Γ[k] = sites[k].Γ
        Λ[k] = sites[k].Λ1
    end
    Λ[N+1] = sites[N].Λ2
	return Γ, Λ 
end

function randomGenericSite(Dl,d,Dr, T = ComplexF64; purification = false)
	Γ = rand(T,Dl,d,Dr)
	return GenericSite(Γ/norm(Γ), purification)
end

function randomLeftOrthogonalSite(Dl,d,Dr, T = ComplexF64; purification = false, method=:qr)
	Γ = randomGenericSite(Dl,d,Dr,T,purification=purification)
	return to_left_orthogonal(Γ, method=method)[1]
end

function randomRightOrthogonalSite(Dl,d,Dr, T = ComplexF64; purification = false, method=:qr)
	Γ = randomGenericSite(Dl,d,Dr,T,purification=purification)
	return to_right_orthogonal(Γ, method=method)[1]
end

function randomOrthogonalLinkSite(Dl,d,Dr, T = ComplexF64; purification = false)
	Γ = randomGenericSite(Dl,d,Dr,T,purification=purification)
	_, ΛL0,_ = svd(Γ, :leftorthogonal)
	_,_,ΓR = svd(Γ, :rightorthogonal)
	ΛL = ΛL0/norm(ΛL0)
	R = ΛL*ΓR
	U, ΛR,_ = svd(R, :leftorthogonal)
	final = inv(ΛL)*U
	return OrthogonalLinkSite(final, ΛL, ΛR)
end
