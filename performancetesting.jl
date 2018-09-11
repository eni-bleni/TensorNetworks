using MPS

## We know how certain operations should scale with
## bond dimension and system size. We should test this.
time = []
Ds = []
mpo = MPS.IsingMPO(10,1,1,0)
mpo0 = MPS.IsingMPO(10,1,1,0)
for D = 1:9
    # mps = MPS.randomMPS(10,2,)
    # @time MPS.MPSnorm(mps)
    println(D)
    push!(time, @elapsed MPS.traceMPOprod(mpo,mpo0,2))
    push!(Ds,first(size(mpo[5])))
    mpo = MPS.addmpos(mpo,mpo,false)
end
a,b = linreg(log.(Ds),log.(time))
println(b)
