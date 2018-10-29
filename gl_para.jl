# using Plots
# nbr_of_workers = 5
filepath = string(@__DIR__,"\\MPSmodule.jl")
# addedprocs=addprocs(nbr_of_workers- length(workers()))
include(filepath)
# @sync @parallel for p in addedprocs
#     remotecall_wait(include,p,filepath)
# end
using MPS

N=100
maxD=50
tol=1e-20
inc=1
steps = 100
time=1
J0=1
h0=1
g0=0
q = 2*pi*(3/(N-1))
hamblocksTH(time) = MPS.isingHamBlocks(N,J0,h0,g0)
hamblocks(time) = MPS.isingHamBlocks(N,J0,h0,g0)
opEmpo = MPS.IsingMPO(N,J0,h0,g0)
opE(time,g,l) = MPS.gl_mpoExp(g,l,opEmpo)
opmag(time,g,l) = MPS.localOpExp(g,l,sx,Int(floor(N/2)))
opnorm(time,g,l) = MPS.gl_mpoExp(g,l,MPS.IdentityMPO(N,2))
ops = [opE opmag opnorm]
mpo = MPS.IdentityMPO(N,2)
# mps = mpo_to_mps(mpo)
mps = MPS.randomMPS(N,2,5)
g,l = MPS.prepareGL(mpo,maxD)
pert_ops = fill(expm(1e-3*im*sx),N)
# pert_ops = [expm(1e-3*sx*im*x) for x in sin.(q*(-1+(1:N)))]
@time opvalsth, errth = MPS.gl_tebd(g,l,hamblocksTH,-2*im,100,maxD,ops,tol=tol,increment=inc,st2=false)

MPS.ops_on_gl(g,l,pert_ops)
@time opvals, err = MPS.gl_tebd(g,l,hamblocks,time,steps,maxD,ops,tol=tol,increment=inc,st2=false)
