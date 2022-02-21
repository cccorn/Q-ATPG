import qiskit
import numpy as np
import warnings
import pickle
from lib.libspd import SPD_gen
from lib.libsampling import sampling

def genqft(qubits):
    cir = qiskit.QuantumCircuit(qubits)
    for q in range(qubits):
        cir.h(q)
        for tar in range(q+1,qubits):
            theta=np.pi/2**(tar-q)
            cir.cu1(theta,q,tar)
    return cir

def translate_cu1(qc):
    qc=qc.copy()
    qc2=qiskit.QuantumCircuit(qc.num_qubits)
    for ii in range(len(qc)):
        if qc[ii][0].name=='cu1':
            tmp=qc[ii]
            theta=tmp[0].params[0]
            idx1=tmp[1][0].index
            idx2=tmp[1][1].index
            qc2.rz(theta/2.0,idx1)
            qc2.rz(theta/2.0,idx2)
            qc2.cx(idx1,idx2)
            qc2.rz(-theta/2.0,idx2)
            qc2.cx(idx1,idx2)
        else:
            qc2.data.append(qc[ii])
    return qc2

if __name__=='__main__':
    warnings.filterwarnings('ignore')

    n=3
    qc=genqft(n)
    qc=translate_cu1(qc)
    ii=np.random.randint(len(qc))
    print('Consider the missing-gate fault on gate '+str(ii))

    #SPD generation
    print('SPD Generation:')
    num_qubits=qc[ii][0].num_qubits
    spd_rho,spd_M,success_prob=SPD_gen(qc,ii,np.eye(2**num_qubits))
    with open('./results/qft'+str(n)+'_'+str(ii)+'.spd','wb') as f:
        pickle.dump([spd_rho,spd_M,success_prob],f)

    #Sampling
    fault=np.random.randint(2)
    if fault==1:
        qc.data.pop(ii)
        expected=1-success_prob
    else:
        expected=success_prob
    print('Sampling:')
    epsilon=0.3
    delta=0.3
    m=sampling(spd_rho,spd_M,qc,epsilon,delta)
    
    print('Expected:'+str(expected))
    print('Estimation:'+str(m))
