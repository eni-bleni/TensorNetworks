# using Plots
# nbr_of_workers = 5
filepath = string(@__DIR__,"\\MPSmodule.jl")
# addedprocs=addprocs(nbr_of_workers- length(workers()))
include(filepath)
# @sync @parallel for p in addedprocs
#     remotecall_wait(include,p,filepath)
# end
using LinearAlgebra
using Main.MPS
# using Traceur
BLAS.set_num_threads(1)
N=20
maxD=10
tol=1e-12
inc=1
steps = 400
tot_time=1
J0=1
h0=1
g0=0
q = 2*pi*(3/(N-1))
hamblocksTH(time) = MPS.isingHamBlocks(N,J0,h0,g0)
hamblocks(time) = MPS.isingHamBlocks(N,J0,h0,g0)
opEmpo = MPS.IsingMPO(N,J0,h0,g0)
opE(time,g,l) = MPS.gl_mpoExp(g,l,opEmpo)
opmag(time,g,l) = MPS.localOpExp(g,l,sx,Int(floor(N/2)))
glnorm(time,g,l) = MPS.gl_mpoExp(g,l,MPS.IdentityMPO(N,2))
ops = [opE opmag glnorm]
mpo = MPS.IdentityMPO(N,2)
# mps = mpo_to_mps(mpo)
mps = MPS.randomMPS(N,2,5)
g,l = MPS.prepareGL(mpo,maxD)
# pert_ops = [expm(1e-3*sx*im*x) for x in sin.(q*(-1+(1:N)))]
@time opvalsth, errth = MPS.gl_tebd(g,l,hamblocksTH,-2*im,20,maxD,ops,tol=tol,increment=inc,st2=true)
gA = deepcopy(g)
lA = deepcopy(l)
gB = deepcopy(g)
lB = deepcopy(l)
# pert_ops = fill(exp(1e-3*im*sx),N)

pert_ops = fill(ComplexF64.(si),N)
pert_ops[Int(floor(N/2))] = exp(1e-3*im*sx)
# pert_ops[Int(floor(N/2))] = exp(1e-2*im*sx)

# MPS.ops_on_gl(g,l,pert_ops)
# pert_ops = fill(exp(1e-3*im*sx),N)
# pert_mpo = MPS.translationMPO(N,sx)
# pertg = MPS.mpo_on_gl(gA,lA,pert_mpo)
# gA, lA = MPS.prepareGL(pertg,maxD,tol)

# MPS.ops_on_gl(gA,lA,pert_ops)
# pert_ops = fill(Complex128.(si),N)
# MPS.gl_ct!(gB)
pert_ops = fill(ComplexF64.(si),N)
pert_ops[Int(floor(N/2))] = sx
# MPS.ops_on_gl(gA,lA,pert_ops)
# MPS.ops_on_gl(gB,lB,pert_ops)
opvalsconst, err = MPS.gl_tebd(g,l,hamblocks,tot_time,steps,maxD,ops,tol=tol,increment=inc,st2=true)
# @time opvals2, errA,errB, times = MPS.gl_tebd_c(gA,lA,gB,lB,hamblocks,tot_time,steps,maxD,tol=tol,increment=inc,st2=true)
