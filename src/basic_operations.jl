function reverse_direction(dir::Symbol)
	if dir ==:right
		return :left
	else
		if dir!==:left
			@warn "No direction chosen in reverse_direction. Defaulting to :left"
		end
		return :right
	end
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
		S = S ./ norm(S)
	end
	D = length(S)
	@views return F.U[:, 1:D], S, F.Vt[1:D, :], D,err
end


"""
	split_truncate(tensor, args)

Split and truncate a two-site tensor
"""
function split_truncate!(theta, args::TruncationArgs)
	#D1l,d,d,D2r = size(theta)
    #theta = reshape(theta, D1l*d,d*D2r)
	F = try
		# println(real(theta))
        svd!(theta)
    catch y
        svd!(theta,alg=LinearAlgebra.QRIteration())
    end
    U,S,Vt,Dm,err = truncate_svd(F, args)
    return U,S,Vt,Dm,real(err)
end


function isleftcanonical(data)
	s = size(data)
	m = reshape(data,s[1]*s[2],s[3])
	id = m'*m
	#@tensor id[:] := conj(data[1,2,-1])*data[1,2,-2]
	return id ≈ one(id)
end  
function isrightcanonical(data)
	s = size(data)
	m = reshape(data,s[1],s[2]*s[3])
	id = m*m'
	#@tensor id[:] := conj(data[-1,2,1])*data[-2,2,1]
	return id ≈ one(id)
end 
