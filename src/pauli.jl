
const sx = [0 1; 1 0]
const sy = [0 -1im; 1im 0]
const sz = [1 0; 0 -1]
const si = [1 0; 0 1]
const s0 = [0 0; 0 0]
const ZZ = kron(sz, sz)
const ZI = kron(si, sz)
const IZ = kron(sz, si)
const XI = kron(si, sx)
const IX = kron(sx, si)
const XY = kron(sy, sx)
const YX = kron(sx, sy)
const II = kron(si, si)
