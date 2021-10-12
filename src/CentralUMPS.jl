function CentralUMPS(mps::UMPS)
    mps = canonicalize(mps)
    n = length(mps)
    ΓL = [GenericSite(mps[k],:left) for k in 1:n]
    ΓR = [GenericSite(mps[k],:right) for k in 1:n]
    CentralUMPS(ΓL,ΓR,mps.Λ[1],ispurification(mps),mps.truncation,mps.error)
end

function transfer_matrix(mps::CentralUMPS; half=:left)
    if half==:left
        return transfer_matrix(mps.ΓL,:right)
    elseif half==:right
        return transfer_matrix(mps.ΓR, :left)
    else
        throw("Choose direction :left or :right")
    end
end

function regularize(tm)
    idR = vec(ones(eltype(tm),size(tm,2)))
    idL = vec(ones(eltype(tm),size(tm,2)))
    vals, vecs, info = eigsolve(tm,idR,2)
    vals2, vecs2, info2 = eigsolve(tm',idL,2)
    @assert vals[1] ≈ vals2[1] ≈ 1 "Dominant eigenvalue not equal to 1"
    lr = vecs2[1]'*vecs[1]
    reg(v) = tm*v - vecs[1]*(vecs2[1]'*v)/lr
    LinearMap{eltype(tm)}(reg,size(tm,1), size(tm,2))
end