
##-------------------------  analytic tensors for binary MERA ---------------------------------

""" returns the isometry w for the critical Ising model, based on wavelets (Vidal,Evenbly)"""
function constr_w()
    w = (sqrt(3)+sqrt(2))/4*II + (sqrt(3)-sqrt(2))/4*ZZ + im*(1-sqrt(2))/4*YX + im*(1+sqrt(2))/4*XY
    w = reshape(w, (2,2,2,2))
    w = w[:,1,:,:]
    ## satisfies: @tensor id[-1,-2]:=w[-1,2,3]*conj(w[-2,2,3])
    return w
end

""" returns the disentangler u for the critical Ising model"""
function constr_u()
    u = (sqrt(3)+2)/4*II + (sqrt(3)-2)/4*ZZ + im/4*YX + im/4*XY
    u = reshape(u, (2,2,2,2))
    ## satisfies: @tensor id[-1,-2,-3,-4]:=u[-1,-2,3,4]*conj(u[-3,-4,3,4])
    return u
end

""" returns the dim=2 isometry w and disentangler u for the critical Ising model"""
function constr_wu_dim2(type=ComplexF64)
    w = constr_w_theta(type(pi)/12)
    u = constr_u_theta(-type(pi)/6)
    return w, u
end

""" returns the dim=8 isometry w and disentangler u for the critical Ising model"""
function constr_wu_dim8()
    ### eqn. (B.3),(B.4),(C.15) in [1602.01166]
    theta1 = atan((56+14*sqrt(106)-sqrt(7)*(23+2*sqrt(106))*sqrt(4*sqrt(106)-39))/105)
    theta2 = atan((17*sqrt(7)+2*sqrt(742))*sqrt(4*sqrt(106)-39)/105)
    theta3 = atan(-sqrt(7(4*sqrt(106)-39))/35)
    theta4 = -pi/2
    w1 = constr_w_theta(theta1)
    u2 = constr_u_theta(theta2)
    u3 = constr_u_theta(theta3)
    u4 = constr_u_theta(theta4)

    @tensoropt w[-1,-2,-3,-4,-5,-6,-7,-8,-9] := w1[-1,-4,1]*w1[-2,2,3]*w1[-3,4,-9]*u2[1,2,-5,5]*u2[3,4,6,-8]*u3[5,6,-6,-7]
    @tensoropt u[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12] := u2[-3,-4,1,2]*u3[-2,1,3,4]*u3[2,-5,5,6]*u4[-1,3,-7,-8]*u4[4,5,-9,-10]*u4[6,-6,-11,-12]
    w = reshape(w, 8,8,8)
    u = reshape(u, 8,8,8,8)

    return w, u
end

""" returns the general isometry w for the critical Ising model"""
function constr_w_theta(theta)
    w = (sqrt(2)*cos(theta-pi/4)+1)/(2*sqrt(2))*II + (sqrt(2)*cos(theta-pi/4)-1)/(2*sqrt(2))*ZZ + im*(1-sqrt(2)*sin(theta-pi/4))/(2*sqrt(2))*XY - im*(1+sqrt(2)*sin(theta-pi/4))/(2*sqrt(2))*YX
    w = reshape(w, (2,2,2,2))
    w = w[:,1,:,:]
    ## satisfies: @tensor id[-1,-2]:=w[-1,2,3]*conj(w[-2,2,3])
    return w
end

""" returns the general disentangler u for the critical Ising model"""
function constr_u_theta(theta)
    u = (cos(theta)+1)/2*II + (cos(theta)-1)/2*ZZ - im*sin(theta)/2*YX - im*sin(theta)/2*XY
    u = reshape(u, (2,2,2,2))
    ## satisfies: @tensor id[-1,-2,-3,-4]:=u[-1,-2,3,4]*conj(u[-3,-4,3,4])
    return u
end

""" returns the 2-site scaling superoperator of the binary MERA"""
function S2_binary(w,u) # does not yet include 2->3 site operators
    @tensoropt S2[-1,-2,-3,-4,-5,-6,-7,-8] := w[-1,1,2]*u[2,3,-5,-6]*w[-2,3,4]*conj(w[-3,1,5])*conj(u[5,6,-7,-8])*conj(w[-4,6,4])
    sS2 = size(S2)
    S2 = reshape(S2, sS2[1]*sS2[2]*sS2[3]*sS2[4], sS2[5]*sS2[6]*sS2[7]*sS2[8])
    return S2
end

""" returns the nonlocal 2-site scaling superoperator of the binary MERA"""
function S2nonlocal_binary(w,u)
    @tensoropt S2[-1,-2,-3,-4,-5,-6,-7,-8] := sz[1,7]*w[-1,1,2]*u[2,3,-5,-6]*w[-2,3,4]*conj(w[-3,7,5])*conj(u[5,6,-7,-8])*conj(w[-4,6,4])
    sS2 = size(S2)
    S2 = reshape(S2, sS2[1]*sS2[2]*sS2[3]*sS2[4], sS2[5]*sS2[6]*sS2[7]*sS2[8])
    return S2
end

""" returns the 3-site scaling superoperator of the binary MERA"""
function S3_binary(w,u)
    @tensoropt S3[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12] := 0.5*(
        w[-1,2,3]*u[3,4,5,-7]*w[-2,4,6]*u[6,7,-8,-9]*w[-3,7,8]*conj(w[-4,2,9])*conj(u[9,10,5,-10])*conj(w[-5,10,11])*conj(u[11,12,-11,-12])*conj(w[-6,12,8]) +
        w[-1,2,3]*u[3,4,-7,-8]*w[-2,4,6]*u[6,7,-9,5]*w[-3,7,8]*conj(w[-4,2,9])*conj(u[9,10,-10,-11])*conj(w[-5,10,11])*conj(u[11,12,-12,5])*conj(w[-6,12,8])  )
    sS3 = size(S3)
    S3 = reshape(S3, sS3[1]*sS3[2]*sS3[3]*sS3[4]*sS3[5]*sS3[6], sS3[7]*sS3[8]*sS3[9]*sS3[10]*sS3[11]*sS3[12])
    return S3
end

""" returns the nonlocal 3-site scaling superoperator of the binary MERA"""
function S3nonlocal_binary(w,u)
    @tensoropt S3[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12] := 0.5*(
        sz[2,22]*w[-1,2,3]*u[3,4,5,-7]*w[-2,4,6]*u[6,7,-8,-9]*w[-3,7,8]*conj(w[-4,22,9])*sz[5,52]*conj(u[9,10,52,-10])*conj(w[-5,10,11])*conj(u[11,12,-11,-12])*conj(w[-6,12,8]) +
        sz[2,22]*w[-1,2,3]*u[3,4,-7,-8]*w[-2,4,6]*u[6,7,-9,5]*w[-3,7,8]*conj(w[-4,22,9])*conj(u[9,10,-10,-11])*conj(w[-5,10,11])*conj(u[11,12,-12,5])*conj(w[-6,12,8])  )
    sS3 = size(S3)
    S3 = reshape(S3, sS3[1]*sS3[2]*sS3[3]*sS3[4]*sS3[5]*sS3[6], sS3[7]*sS3[8]*sS3[9]*sS3[10]*sS3[11]*sS3[12])
    return S3
end

""" apply S3 to a vector "v" and return resulting vector S3v"""
function apply_S3_to_vector(w,u,d,v)
    v = reshape(v, d,d,d,d,d,d) # 3-site gate
    @tensoropt S3v_1[-1,-2,-3,-4,-5,-6] := w[-1,1,2]*w[-2,3,6]*w[-3,7,10]*u[2,3,4,5]*u[6,7,8,9]*v[5,8,9,13,16,17]*conj(u[11,12,4,13])*conj(u[14,15,16,17])*conj(w[-4,1,11])*conj(w[-5,12,14])*conj(w[-6,15,10])
    @tensoropt S3v_2[-1,-2,-3,-4,-5,-6] := w[-1,1,2]*w[-2,3,6]*w[-3,7,10]*u[2,3,4,5]*u[6,7,8,9]*v[4,5,8,13,16,17]*conj(u[11,12,13,16])*conj(u[14,15,17,9])*conj(w[-4,1,11])*conj(w[-5,12,14])*conj(w[-6,15,10])
    S3v = 0.5*(S3v_1 + S3v_2)
    return reshape(S3v, d^6)
end
