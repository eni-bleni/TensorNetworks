gates = isingHamGates(3,1,0,0);
mpo = IsingMPO(3,1,0,0);

mps = canonicalize(randomUMPS(ComplexF64,2,2,1));
transfer_matrix(mps);
expectation_value(mps,gates[2],1);

mps = canonicalize(randomOrthOpenMPS(ComplexF64,2,2,1),1)
expectation_value(mps,gates[2],1);

mps = canonicalize(randomOpenMPS(ComplexF64,2,2,1))
expectation_value(mps,gates[2],1);