from .libspd import *





def sampling(spd_rho,spd_M,CUT,delta,epsilon):
    cost_rho=spd_rho.norm1_c_w()
    cost_M=spd_M.norm1_c()

    t=2/(delta**2)
    t*=np.log(2/epsilon)
    t*=(cost_rho**2)
    t*=(cost_M**2)
    
    p_rho=np.array([abs(spd_rho.cs[ii])*spd_rho.As[ii].rank for ii in range(len(spd_rho.As))])/cost_rho
    p_M=np.array([abs(spd_M.cs[ii]) for ii in range(len(spd_M.As))])/cost_M

    simulator = qiskit.Aer.get_backend('aer_simulator')
    
    count=0
    
    for tau in tqdm(range(int(t)+1)):
        ii=np.random.choice(np.arange(len(p_rho)),p=p_rho)
        jj=np.random.choice(np.arange(len(p_M)),p=p_M)
        
        A=spd_rho.As[ii]
        B=spd_M.As[jj]
        
        U_A=gen_cliff([4**ii for ii in range(len(A.paulis))],A.paulis,[0]*len(A.paulis),A.phases,A.n)
        U_B=gen_cliff(B.paulis,[4**ii for ii in range(len(B.paulis))],B.phases,[0]*len(B.paulis),B.n)
        
        init=qiskit.QuantumCircuit(A.n)
        m=len(A.paulis)
        ll=np.random.randint(0,2,A.n-m)
        for kk in range(A.n-m):
            if ll[kk]==1:
                init.x(m+kk)

        U_A=qiskit.quantum_info.decompose_clifford(U_A,'greedy')
        U_B=qiskit.quantum_info.decompose_clifford(U_B,'greedy')
        qc=qiskit.QuantumCircuit(A.n,len(B.paulis))
        qc=qiskit.QuantumCircuit.compose(qc,init)
        qc=qiskit.QuantumCircuit.compose(qc,U_A)
        qc=qiskit.QuantumCircuit.compose(qc,CUT)
        qc=qiskit.QuantumCircuit.compose(qc,U_B)
        qc.measure([kk for kk in range(len(B.paulis))],[kk for kk in range(len(B.paulis))])
        
        qc = qiskit.transpile(qc,simulator)
        result = simulator.run(qc, shots=1, memory=True).result()
        memory = result.get_memory(qc)
        if memory[0]==('0'*len(B.paulis)):
            count=count+np.sign(spd_rho.cs[ii]*spd_M.cs[jj])*cost_rho*cost_M

    return count/t
