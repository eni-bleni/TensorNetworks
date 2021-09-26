abstract type AbstractMixer end

struct ShiftCenter <: AbstractMixer end

mutable struct SubspaceExpand <: AbstractMixer
    alpha::Float64
    rate::Float64
    oldmin::Union{Float64,Nothing}
    function SubspaceExpand(alpha,rate) 
        @assert rate < 1; 
        return new(alpha, rate, nothing)
    end
end
SubspaceExpand(alpha) = SubspaceExpand(alpha,3/4)

function shift_center!(mps,j,dir,::ShiftCenter; kwargs...)
    if dir==:right 
        shift_center_right!(mps)
    elseif dir==:left
        shift_center_left!(mps)
    end
end

function shift_center!(mps,j,dir,SE::SubspaceExpand; mpo,env, kwargs...)
    newmin = transpose(transfer_matrix(mps[j]', mpo[j], mps[j]) * vec(env.R[j])) * vec(env.L[j])
    if dir==:right
        dirval=+1
        j1 = j
        j2 = j+1
    elseif dir==:left
        dirval=-1
        j1 = j-1
        j2 = j
    end

    A,B = subspace_expand(SE.alpha,mps[j],mps[j+dirval],env[j, reverse_direction(dir)], mpo[j], mps.truncation, dir)
    mps.center+=dirval
    mps.Γ[j] = A
    mps.Γ[j+dirval] = B
    T = transfer_matrix(adjoint.(mps[j1:j2]), mpo[j1:j2],mps[j1:j2], :left)
    truncmin = transpose( T*vec(env.R[j2])) * vec(env.L[j1])
    
    if SE.oldmin !== nothing
        if abs((truncmin - newmin)/(SE.oldmin - newmin)) >.3 
            SE.alpha *= SE.rate
        else
            SE.alpha *= 1/SE.rate
        end
    end
    SE.oldmin = real.(truncmin)
end

function iterative_compression(target::AbstractMPS, guess::AbstractMPS, prec=1e-8; maxiter = 50, shifter=ShiftCenter)
    env = environment(guess',target)
    # mps = guess
#    mps = iscanonical(guess) ? guess :  canonicalize(guess)
    mps = canonicalize(guess)
    
    set_center!(mps,1)
    dir = :right
    targetnorm = norm(target)
    IL(site) = Array(vec(Diagonal(1.0I,size(site,1)))) 
    IR(site) = Array(vec(Diagonal(1.0I,size(site,3))))
    errorfunc(mps) = 1 - abs(scalar_product(target,mps))#real(targetnorm - sum([transpose(transfer_matrix(site) * IR(site))*IL(site) for site in mps[1:end]]))
    count=1
    error = errorfunc(mps)
    error = error 
    println(error)
    while error > prec && count<maxiter
        mps, env = sweep(target,mps,env,dir, prec)
        newerror = errorfunc(mps)
        if abs(error-newerror)<prec
            break
        end
        error = newerror
        println(error)
        dir = reverse_direction(dir)
    end
    return mps
end

function sweep(target,mps,env,dir, prec;kwargs...)
    L = length(mps)
    shifter = get(kwargs, :shifter, ShiftCenter())
    if dir==:right 
        itr = 1:L-1 + isinfinite(mps)
        dirval=1
    elseif dir==:left
        itr = L:-1:2 - isinfinite(mps)
        dirval=-1
    else
        @error "In sweep: choose dir :left or :right"
    end
    for j in itr
        @assert (iscenter(mps,j)) "The optimization step is not performed at the center of the mps: $(center(mps)) vs $j"
        newsite = local_mul(env.L[j],env.R[j],target[j])
        mps[j] = newsite/norm(newsite)
        shift_center!(mps,j,dir,shifter; error = error)
        update! = dir==:right ? update_left_environment! : update_right_environment!
        update!(env,j,mps[j]',target[j])
    end
    return mps, env
end
