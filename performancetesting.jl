using MPS
using TensorOperations
## We know how certain operations should scale with
## bond dimension and system size. We should test this.
# time = []
# Ds = []
# mpo = MPS.IsingMPO(10,1,1,0)
# mpo0 = MPS.IsingMPO(10,1,1,0)
# for D = 1:9
#     # mps = MPS.randomMPS(10,2,)
#     # @time MPS.MPSnorm(mps)
#     println(D)
#     push!(time, @elapsed MPS.traceMPOprod(mpo,mpo0,2))
#     push!(Ds,first(size(mpo[5])))
#     mpo = MPS.addmpos(mpo,mpo,false)
# end
# a,b = linreg(log.(Ds),log.(time))
# println(b)

#testing parallel for
function loop(n)
        t = Array{Any}(n)
        i = 0
    for k = 2.^(3:n)
        i+=1
        l = diagm(rand(k))
        l2 = diagm(rand(k))
        g = rand(k,2,2,k)
        block = rand(2,2,2,2)
        sg=size(g)
        # @time gm=reshape(g,sg[1],sg[2]*sg[3])
        # t[i] = @elapsed theta = l*gm
        # @time theta = reshape(theta,sg[1],sg[2],sg[3])
        # t[i] = @elapsed inv(l)
        # U=rand(k,2,k)
        # V=rand(k,2,2,k)
        M=rand(Complex128,k,k)
        function lin(v)
                @tensor r[:] = M[-1,1]*v[1]
                return r
        end
        function linc(v)
                @tensor r[:] = M'[-1,1]*v[1]
                return r
        end
        t[i]= @elapsed svd(M)
        linmap = LinearMap{Complex128}(lin,linc,k,k)
        t[i]/=@elapsed svds(M,nsv=min(100,Int(k/2)))

        # t[i] = @elapsed il = spdiagm(1./diag(l))
        # t[i] += @elapsed il2 = spdiagm(1./diag(l2))
        # t[i] = @elapsed U = sparse_l(U,l,:left)
        # t[i] += @elapsed U = sparse_l(U,l2,:right)
        # t[i] -= @elapsed @tensor TL2[:] := l[-1,1]*U[1,-2,4]*l2[4,-4]
        # t[i] -= @elapsed @tensor TR2[:] := V[-1,-2,-3,3]*inv(l)[3,-4]

        # t[i] = @elapsed @tensor theta2[:] := l[-1,1]*g[1,-2,-3] #*l[4,-3]
        # t[i]= t[i]+ @elapsed @tensor theta[:] := theta[-1,2,5]*g[5,3,6]*l[6,-4]*block[-3,-2,3,2]
        # @time @tensor theta[:] := lL[-1,1]*gL[1,2,4]*lM[4,5]*gR[5,3,6]*lR[6,-4]*block[-2,-3,2,3]
        # st = size(theta)
        # t[i] = @elapsed svd(reshape(theta,st[1]*st[2],st[3]*st[4]))
        # function r(a)
        #         return rand(Complex128,4,4)
        # end
        # hb = r.(1:k)
        # t[i] = @elapsed reshape.(expm.(-1im*hb),2,2,2,2)
        #
        # @inbounds @simd for j = 1:k
        #         t[i] -= @elapsed reshape(expm(-1im*hb[j]),2,2,2,2)
        # end

        println(k,"_ ",t[i])
    end
    return t
end
