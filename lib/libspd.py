import qiskit
import numpy as np
import cvxpy as cp
import os
import pickle
import copy
from tqdm.auto import tqdm
from . import config





def gf2_rank(rows):
    rows=rows.copy()
    rank = 0
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    return rank





def gf2_indp(rows):
    rows=rows.copy()
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
        else:
            return False
    return True





def gf2_find_indp(rows):
    rows=rows.copy()
    out=[]
    idx=len(rows)-1
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            out.append(idx)
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
        idx-=1
    return out





def gf2_find_decompose(basis,v):
    n=len(basis)
    basis=basis.copy()
    P=[2**(n-1-ii) for ii in range(n)]
    out=0
    idx=len(basis)-1
    while basis:
        pivot_row = basis.pop()
        lsb = pivot_row & -pivot_row
        if lsb & v:
            out=out ^ P[idx]
            v = v ^ pivot_row
        for index, row in enumerate(basis):
            if row & lsb:
                basis[index] = row ^ pivot_row
                P[index]=P[index] ^ P[idx]
        idx-=1
    if v==0:
        return out
    else:
        return False





def int2bin(x):
    if x==0:
        return []
    result=[]
    while(True):
        if x==0:
            break
        else:
            result.insert(0,x%2)
        x=x//2
    return result





def bin2int(x):
    n=len(x)
    result=[0 for ii in range(n)]
    tmp1=1
    result=0
    for ii in range(n):
        result+=tmp1*int(x[n-1-ii])
        tmp1*=2
    return result





def inner_product(x,y):
    z=x&y
    a=int2bin(z)
    return sum(a)%2





def is_commute(x,y):
    a=int2bin(x)
    b=int2bin(y)
    if len(a)%2==1:
        a.insert(0,0)
    if len(b)%2==1:
        b.insert(0,0)
    result=0
    n=min(len(a),len(b))
    for ii in range(n//2):
        result+=(a[-1-2*ii]*b[-1-(2*ii+1)]+a[-1-(2*ii+1)]*b[-1-2*ii])
    return (not result%2)





def enumerate_isotropic_dim(n,m):
    assert n>=m
    print('cache:'+str(n)+'_'+str(m))
    N=4**n
    landmarks=[ii for ii in range(m)]
    result=[]

    total_num=int(iso(n,m))
    pbar = tqdm(total=total_num)
    count = 0
    while count<total_num:
        ky=1
        if gf2_indp(landmarks):
            for ii in range(m):
                for jj in range(ii+1,m):
                    if not is_commute(landmarks[ii],landmarks[jj]):
                        ky=0
                        break
                if ky==0:
                    break
        else:
            ky=0
        if ky==1:
            for w in result:
                cf=1
                for ii in range(m):
                    if gf2_indp(w+[landmarks[ii]]):
                        cf=0
                        break
                if cf==1:
                    ky=0
                    break
        if ky==1:
            result.append(landmarks.copy())
            pbar.update(1)
            count+=1
            
        daotoule=m-1
        for ii in range(m):
            if landmarks[m-1-ii]==N-1-ii:
                daotoule-=1
            else:
                break
        if daotoule>=0:
            landmarks[daotoule]+=1
            for ii in range(daotoule+1,m):
                landmarks[ii]=landmarks[daotoule]+(ii-daotoule)
        else:
            break
    pbar.close()
    return result





def iso(n,m):
    result=1
    for ii in range(m):
        result*=((4**(n-ii)-1)/(2**(m-ii)-1))
    return result





def iso_all(n):
    result=0
    for ii in range(n+1):
        result+=iso(n,ii)
    return result





def infer_phase(pauli1,pauli2):
    pauli1=int2bin(pauli1)
    pauli2=int2bin(pauli2)
    n=((max(len(pauli1),len(pauli2))+1)//2)*2
    pauli1=[0]*(n-len(pauli1))+pauli1
    pauli2=[0]*(n-len(pauli2))+pauli2
    phase=0
    for ii in range(n//2):
        if pauli1[ii*2:ii*2+2]==[0,1] and pauli2[ii*2:ii*2+2]==[1,0]:
            #ZX
            phase+=1
        elif pauli1[ii*2:ii*2+2]==[1,0] and pauli2[ii*2:ii*2+2]==[1,1]:
            #XY
            phase+=1
        elif pauli1[ii*2:ii*2+2]==[1,1] and pauli2[ii*2:ii*2+2]==[0,1]:
            #YZ
            phase+=1
        elif pauli1[ii*2:ii*2+2]==[1,0] and pauli2[ii*2:ii*2+2]==[0,1]:
            #XZ
            phase+=3
        elif pauli1[ii*2:ii*2+2]==[1,1] and pauli2[ii*2:ii*2+2]==[1,0]:
            #YX
            phase+=3
        elif pauli1[ii*2:ii*2+2]==[0,1] and pauli2[ii*2:ii*2+2]==[1,1]:
            #ZY
            phase+=3
    return (phase//2)%2





def span_with_sign(basis,sign,n):
    m=len(basis)
    N=2**n
    result=[0 for ii in range(4**n)]
    for ii in range(2**m):
        iii=int2bin(ii)
        iii=[0 for jj in range(m-len(iii))]+iii
        tmp=0
        signsign=0
        for jj in range(m):
            if iii[jj]==1:
                signsign=signsign+sign[jj]+infer_phase(tmp,basis[jj])
                tmp=tmp^(basis[jj])
        signsign=signsign%2
        if signsign==0:
            result[tmp]=1
        else:
            result[tmp]=-1
    return result





def enumerate_sp_dim(n,m):
    isotropic=enumerate_isotropic_dim(n,m)
    doc=[]
    sp=[]
    for subspace in isotropic:
        for sign in range(2**m):
            sign_bin=int2bin(sign)
            sign_bin=[0 for ii in range(m-len(sign_bin))]+sign_bin
            doc.append({'basis':subspace,'phase':sign_bin})
            sp.append(span_with_sign(subspace,sign_bin,n))
    return sp,doc





def enumerate_sp(n,refresh=0):
    if refresh==0:
        if n in config.cache.keys():
            result,result_doc=config.cache[n]
            return result,result_doc
        elif os.path.isfile('./cache/'+str(n)+'.sp'):
            with open('./cache/'+str(n)+'.sp','rb') as f:
                result,result_doc = pickle.load(f)
            config.cache[n]=[result,result_doc]
            return result,result_doc

    result=np.empty((0,4**n))
    result_doc=[]

    for m in range(n+1):
        tmp_sp,tmp_doc=enumerate_sp_dim(n,m)
        tmp_sp=np.array(tmp_sp)
        result=np.vstack((result,tmp_sp*(2**(n-m))))
        result_doc=result_doc+tmp_doc
            
    config.cache[n]=[result,result_doc]
    with open('./cache/'+str(n)+'.sp','wb') as f:
        pickle.dump([result,result_doc],f)

    return result,result_doc





def smtn_normalize_paulis(paulis1,paulis2,phases1,phases2):
    assert len(paulis1)==len(paulis2)
    assert len(paulis1)==len(phases1)
    assert len(phases1)==len(phases2)
    anti_commute=[-1 for ii in range(len(paulis1))]
    paulis1=paulis1.copy()
    paulis2=paulis2.copy()
    phases1=phases1.copy()
    phases2=phases2.copy()
    for ii in range(len(paulis1)):
        if anti_commute[ii]>=0:
            continue
        for jj in range(ii+1,len(paulis1)):
            if anti_commute[jj]>=0:
                continue
            if not is_commute(paulis1[ii],paulis1[jj]):
                for kk in range(jj+1,len(paulis1)):
                    if anti_commute[kk]>=0:
                        continue
                    if not is_commute(paulis1[ii],paulis1[kk]):
                        phases1[kk]=(phases1[kk]+phases1[jj]+infer_phase(paulis1[kk],paulis1[jj]))%2
                        phases2[kk]=(phases2[kk]+phases2[jj]+infer_phase(paulis2[kk],paulis2[jj]))%2
                        paulis1[kk]=paulis1[kk]^paulis1[jj]
                        paulis2[kk]=paulis2[kk]^paulis2[jj]
                for kk in range(ii+1,len(paulis1)):
                    if anti_commute[kk]>=0:
                        continue
                    if not is_commute(paulis1[jj],paulis1[kk]):
                        phases1[kk]=(phases1[kk]+phases1[ii]+infer_phase(paulis1[kk],paulis1[ii]))%2
                        phases2[kk]=(phases2[kk]+phases2[ii]+infer_phase(paulis2[kk],paulis2[ii]))%2
                        paulis1[kk]=paulis1[kk]^paulis1[ii]
                        paulis2[kk]=paulis2[kk]^paulis2[ii]
                anti_commute[ii]=jj
                anti_commute[jj]=ii
                break
    return anti_commute,paulis1,paulis2,phases1,phases2





def normalize_paulis(paulis1,phases1):
    assert len(paulis1)==len(phases1)
    anti_commute=[-1 for ii in range(len(paulis1))]
    paulis1=paulis1.copy()
    phases1=phases1.copy()
    for ii in range(len(paulis1)):
        if anti_commute[ii]>=0:
            continue
        for jj in range(ii+1,len(paulis1)):
            if anti_commute[jj]>=0:
                continue
            if not is_commute(paulis1[ii],paulis1[jj]):
                for kk in range(jj+1,len(paulis1)):
                    if anti_commute[kk]>=0:
                        continue
                    if not is_commute(paulis1[ii],paulis1[kk]):
                        phases1[kk]=(phases1[kk]+phases1[jj]+infer_phase(paulis1[kk],paulis1[jj]))%2
                        paulis1[kk]=paulis1[kk]^paulis1[jj]
                for kk in range(ii+1,len(paulis1)):
                    if anti_commute[kk]>=0:
                        continue
                    if not is_commute(paulis1[jj],paulis1[kk]):
                        phases1[kk]=(phases1[kk]+phases1[ii]+infer_phase(paulis1[kk],paulis1[ii]))%2
                        paulis1[kk]=paulis1[kk]^paulis1[ii]
                anti_commute[ii]=jj
                anti_commute[jj]=ii
                break
    return anti_commute,paulis1,phases1





def pad_paulis(anti_commute,paulis,n):
    paulis=paulis.copy()
    anti_commute=anti_commute.copy()
    rows=paulis.copy()
    lsbs=[0 for ii in range(2*n)]
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            lsb = pivot_row & -pivot_row
            lsbs[2*n-len(int2bin(lsb))]=1
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    pad_paulis=[]
    for ii in range(2*n):
        if lsbs[ii]==0:
            pad_paulis.append(2**(2*n-1-ii))

    anti_commute=anti_commute + [-1 for ii in range(len(pad_paulis))]
    origin_len=len(paulis)
    paulis=paulis+pad_paulis

    for ii in range(origin_len):
        if anti_commute[ii]==-1:
            continue
        for jj in range(origin_len,len(paulis)):
            if not is_commute(paulis[ii],paulis[jj]):
                paulis[jj]=paulis[jj]^paulis[anti_commute[ii]]
    
    for ii in range(origin_len):
        if anti_commute[ii]>=0:
            continue
        for jj in range(origin_len,len(paulis)):
            if anti_commute[jj]>=0:
                continue
            if not is_commute(paulis[ii],paulis[jj]):
                for kk in range(origin_len,len(paulis)):
                    if kk==jj:
                        continue
                    if not is_commute(paulis[ii],paulis[kk]):
                        paulis[kk]=paulis[kk]^paulis[jj]
                anti_commute[ii]=jj
                anti_commute[jj]=ii
                break

    for ii in range(origin_len,len(paulis)):
        if anti_commute[ii]==-1:
            continue
        for jj in range(origin_len,len(paulis)):
            if not is_commute(paulis[ii],paulis[jj]):
                paulis[jj]=paulis[jj]^paulis[anti_commute[ii]]

    for ii in range(origin_len,len(paulis)):
        if anti_commute[ii]>=0:
            continue
        for jj in range(ii+1,len(paulis)):
            if anti_commute[jj]>=0:
                continue
            if not is_commute(paulis[ii],paulis[jj]):
                for kk in range(jj+1,len(paulis)):
                    if anti_commute[kk]>=0:
                        continue
                    if not is_commute(paulis[ii],paulis[kk]):
                        paulis[kk]=paulis[kk]^paulis[jj]
                for kk in range(ii+1,len(paulis)):
                    if anti_commute[kk]>=0:
                        continue
                    if not is_commute(paulis[jj],paulis[kk]):
                        paulis[kk]=paulis[kk]^paulis[ii]
                anti_commute[ii]=jj
                anti_commute[jj]=ii
                break
    
    return anti_commute,paulis





def pauli_int2str(pauli,n):
    pauli_bin=int2bin(pauli)
    pauli_bin=([0]*(2*n-len(pauli_bin)))+pauli_bin
    out=''
    for ii in range(n):
        if pauli_bin[ii*2:ii*2+2]==[0,0]:
            out+='I'
        elif pauli_bin[ii*2:ii*2+2]==[0,1]:
            out+='Z'
        elif pauli_bin[ii*2:ii*2+2]==[1,0]:
            out+='X'
        elif pauli_bin[ii*2:ii*2+2]==[1,1]:
            out+='Y'
    return out





def pauli_str2int(pauli):
    n=len(pauli)
    pauli_bin=[]
    for ii in range(n):
        if pauli[ii]=='I':
            pauli_bin+=[0,0]
        elif pauli[ii]=='Z':
            pauli_bin+=[0,1]
        elif pauli[ii]=='X':
            pauli_bin+=[1,0]
        elif pauli[ii]=='Y':
            pauli_bin+=[1,1]
    return bin2int(pauli_bin)





def phase_int2str(phase):
    assert phase==0 or phase==1
    if phase==0:
        return '+'
    elif phase==1:
        return '-'





def gen_cliff(paulis1,paulis2,phases1,phases2,n):
    anti_commute,paulis1,paulis2,phases1,phases2=smtn_normalize_paulis(paulis1,paulis2,phases1,phases2)
    anti_commute1,paulis1=pad_paulis(anti_commute,paulis1,n)
    anti_commute2,paulis2=pad_paulis(anti_commute,paulis2,n)
    stabilizer1=[]
    destabilizer1=[]
    stabilizer2=[]
    destabilizer2=[]
    
    phases1=phases1+[0]*(2*n-len(phases1))
    phases2=phases2+[0]*(2*n-len(phases2))
    
    used=[0]*(2*n)
    for ii in range(2*n):
        if used[ii]==1:
            continue
        stabilizer1.append(phase_int2str(phases1[ii])+pauli_int2str(paulis1[ii],n))
        destabilizer1.append(phase_int2str(phases1[anti_commute1[ii]])+pauli_int2str(paulis1[anti_commute1[ii]],n))
        used[ii]=1
        used[anti_commute1[ii]]=1
    used=[0]*(2*n)
    for ii in range(2*n):
        if used[ii]==1:
            continue
        stabilizer2.append(phase_int2str(phases2[ii])+pauli_int2str(paulis2[ii],n))
        destabilizer2.append(phase_int2str(phases2[anti_commute2[ii]])+pauli_int2str(paulis2[anti_commute2[ii]],n))
        used[ii]=1
        used[anti_commute2[ii]]=1

    cliff1=qiskit.quantum_info.Clifford.from_dict({'stabilizer':stabilizer1,'destabilizer':destabilizer1}).adjoint()
    cliff2=qiskit.quantum_info.Clifford.from_dict({'stabilizer':stabilizer2,'destabilizer':destabilizer2})
    cliff=qiskit.quantum_info.Clifford.compose(cliff1,cliff2)
    return cliff





def gf2_null(rows):
    rows=rows.copy()
    n=len(rows)
    idx=n-1
    P=np.zeros((n,n),dtype=bool)
    for ii in range(n):
        P[ii,ii]=1
    null_basis=[]
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
                    P[index,:]=P[idx,:]^P[index,:]
        else:
            null_basis.append(idx)
        idx-=1
    return P[null_basis]





def intersection_with_phase(A,phase):
    A=copy.deepcopy(A)
    phase=copy.deepcopy(phase)
    n=len(A)
    assert n>=1
    out=A[0]
    out_phase=phase[0]
    for ii in range(1,len(A)):
        n1=len(out)
        n2=len(A[ii])
        tmp=gf2_null(out+A[ii])
        atmp2=[]
        atmp2_phase=[]
        btmp2=[]
        btmp2_phase=[]
        for jj in range(tmp.shape[0]):
            tmp3=0
            tmp3_phase=0
            for kk in range(n1):
                if tmp[jj,kk]==1:
                    tmp3_phase=tmp3_phase+out_phase[kk]+infer_phase(tmp3,out[kk])
                    tmp3=tmp3^out[kk]
            atmp2.append(tmp3)
            atmp2_phase.append(tmp3_phase%2)
            
            tmp3=0
            tmp3_phase=0
            for kk in range(n2):
                if tmp[jj,kk+n1]==1:
                    tmp3_phase=tmp3_phase+phase[ii][kk]+infer_phase(tmp3,A[ii][kk])
                    tmp3=tmp3^A[ii][kk]
            btmp2.append(tmp3)
            btmp2_phase.append(tmp3_phase%2)

        ###filtered by phase
        bad=-1
        for jj in range(tmp.shape[0]):
            if btmp2_phase[jj]!=atmp2_phase[jj]:
                if bad==-1:
                    bad=jj
                else:
                    atmp2_phase[jj]=(atmp2_phase[jj]+atmp2_phase[bad]+infer_phase(atmp2[jj],atmp2[bad]))%2
                    atmp2[jj]=atmp2[jj]^atmp2[bad]
        if bad!=-1:
            atmp2.pop(bad)
            atmp2_phase.pop(bad)

        out=atmp2
        out_phase=atmp2_phase
    return out,out_phase





def exploit_loc(A,A_phase,P_U,P_U_phase,n):
    its_A,its_A_phase=intersection_with_phase(A,A_phase)
    assert gf2_indp(its_A+[P_U])
    for ii in range(len(its_A)):
        if not is_commute(P_U,its_A[ii]):
            for jj in range(ii+1,len(its_A)):
                if not is_commute(P_U,its_A[jj]):
                    its_A_phase[jj]=(its_A_phase[jj]+its_A_phase[ii]+infer_phase(its_A[jj],its_A[ii]))%2
                    its_A[jj]=its_A[jj]^its_A[ii]
            its_A.pop(ii)
            its_A_phase.pop(ii)
            break
    phase=its_A_phase+[P_U_phase]
    Z=[4**ii for ii in range(len(its_A)+1)]
    V1=gen_cliff(its_A+[P_U],Z,phase,[0]*len(phase),n)
    n1=len(its_A)

    G=[]
    G_phase=[]
    for ii in range(len(A)):
        G_i=[]
        G_i_phase=[]
        for jj in range(len(A[ii])):
            tmp=qiskit.quantum_info.Pauli(pauli_int2str(A[ii][jj],n))
            tmp2=(tmp.evolve(V1.adjoint())).to_label()
            phase=0
            if tmp2[0]=='-':
                phase=1
                tmp2=tmp2[1:]
            G_i.append(pauli_str2int(tmp2))
            G_i_phase.append(phase)
        for jj in range(len(G_i)):
            for kk in range(len(its_A)):
                tmp=(G_i[jj]//(4**kk))%4
                assert tmp==0 or tmp==1
                if tmp==1:
                    G_i[jj]=G_i[jj]^(4**kk)
            G_i[jj]=G_i[jj]//(4**len(its_A))
        indp=gf2_find_indp(G_i)
        G_i=[G_i[indpjj] for indpjj in indp]
        G_i_phase=[G_i_phase[indpjj] for indpjj in indp]
        G.append(G_i)
        G_phase.append(G_i_phase)
    union_G=[1]
    for ii in range(len(G)):
        union_G+=G[ii]
    indp=gf2_find_indp(union_G)
    h_i=[union_G[indpii] for indpii in indp]
    anti_commute_h,norm_h,noneed =normalize_paulis(h_i,[0]*len(h_i))
    paulis_zx=[0]*len(norm_h)
    z_count=0
    for ii in range(len(anti_commute_h)):
        if anti_commute_h[ii]==-1:
            paulis_zx[ii]=4**(n-n1-1-z_count)
            z_count+=1
        elif paulis_zx[ii]==0:
            paulis_zx[ii]=4**(n-n1-1-z_count)
            paulis_zx[anti_commute_h[ii]]=(4**(n-n1-1-z_count))*2
            z_count+=1
    n2=n-n1-z_count

    V2=gen_cliff(norm_h,paulis_zx,[0]*len(norm_h),[0]*len(norm_h),n-n1)
    if n1>0:
        I=qiskit.quantum_info.Clifford(qiskit.QuantumCircuit(n1))
        V2=qiskit.quantum_info.Clifford.tensor(V2,I)
    V=qiskit.quantum_info.Clifford.compose(V1,V2)
    return V,n1,n2





def Udiscrimination(U1,U2):
    eigvalues,eigvectors=np.linalg.eig(U1.conjugate().transpose() @ U2)
    re=eigvalues.real.copy()
    im=eigvalues.imag.copy()
    n=len(eigvalues)
    ones=np.ones(n)
    zeros=np.zeros(n)
    
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.square(re.T @ x)+cp.square(im.T @ x)),
                    [ones.T @ x == 1, 
                     x >= zeros])
    prob.solve()

    psi=eigvectors @ np.sqrt(x.value)
    psi1=U1@psi
    psi2=U2@psi
    
    ip=psi1.conjugate().transpose()@ psi2
    r=np.abs(ip)
    theta=np.angle(ip)

    tmp1=psi1+np.exp(-1j*theta)*psi2
    tmp2=psi1-np.exp(-1j*theta)*psi2
    tmp11=np.linalg.norm(tmp1)
    tmp22=np.linalg.norm(tmp2)
    omega1=(tmp1/tmp11+tmp2/tmp22)/np.sqrt(2)
    omega2=(tmp1/tmp11-tmp2/tmp22)/np.sqrt(2)

    success=(np.cos(np.pi/4-np.arccos(r)/2))**2

    return psi,omega1,success





class stab_proj:
    def __init__(self, n, paulis, phases):
        assert gf2_indp(paulis)==True
        assert len(phases)==len(paulis)
        self.n=n
        self.rank=2**(n-len(paulis))
        self.paulis=paulis
        self.phases=phases
    
    def apply_clifford(self, cliff):
        for ii in range(len(self.paulis)):
            pauli=pauli_int2str(self.paulis[ii],self.n)
            pauli2=(qiskit.quantum_info.Pauli(pauli)).evolve(cliff.adjoint()).to_label()
            if pauli2[0]=='-':
                self.phases[ii]=(self.phases[ii]+1)%2
                pauli2=pauli2[1:]
            self.paulis[ii]=pauli_str2int(pauli2)

    def trace_pauli(self, pauli):
        if pauli==0:
            return self.rank
        else:
            decompose=gf2_find_decompose(self.paulis,pauli)
            if decompose==False:
                return 0
            else:
                tmp_phase=0
                tmp_pauli=0
                decompose=int2bin(decompose)
                decompose=[0]*(len(self.paulis)-len(decompose))+decompose
                for ii in range(len(self.paulis)):
                    if decompose[ii]==1:
                        tmp_phase=(tmp_phase+self.phases[ii]+infer_phase(tmp_pauli,self.paulis[ii]))%2
                        tmp_pauli=tmp_pauli^self.paulis[ii]
                assert tmp_pauli==pauli
                if tmp_phase==1:
                    return -self.rank
                else:
                    return self.rank
    
    def is_commute(self, pauli):
        for ii in range(len(self.paulis)):
            if not is_commute(self.paulis[ii],pauli):
                return False
        return True

    def copy(self):
        out=stab_proj(self.n,self.paulis.copy(),self.phases.copy())
        return out
    
    def baoli_truncate(self,truncate_n):
        for ii in range(len(self.paulis)):
            self.paulis[ii]=self.paulis[ii]//(4**truncate_n)
        idx=gf2_find_indp(self.paulis)
        tmp_paulis=[]
        tmp_phases=[]
        for ii in idx:
            tmp_paulis.append(self.paulis[ii])
            tmp_phases.append(self.phases[ii])
        self.paulis=tmp_paulis
        self.phases=tmp_phases
        self.n=self.n-truncate_n
        self.rank=2**(self.n-len(self.paulis))
        
    def tensor_stab_proj(self,A):
        tensor_n=A.n
        for ii in range(len(self.paulis)):
            self.paulis[ii]=self.paulis[ii]*(4**tensor_n)
        self.paulis=self.paulis+A.paulis
        self.phases=self.phases+A.phases
        self.n=self.n+tensor_n
        self.rank=2**(self.n-len(self.paulis))
        
    def pad_identity(self,current_idx,n):
        assert len(current_idx)==self.n
        for ii in range(len(self.paulis)):
            tmp=np.zeros(2*n,dtype=bool)
            tmp2=int2bin(self.paulis[ii])
            for jj in range(len(tmp2)):
                kk=2*current_idx[jj//2]+(jj%2)
                tmp[len(tmp)-1-kk]=tmp2[len(tmp2)-1-jj]
            self.paulis[ii]=bin2int(tmp)
        self.n=n
        self.rank=2**(self.n-len(self.paulis))
        
    def __str__(self):
        tmp_paulis=[]
        for ii in range(len(self.paulis)):
            tmp_pauli='+' if self.phases[ii]==0 else '-'
            tmp_pauli+=pauli_int2str(self.paulis[ii],self.n)
            tmp_paulis.append(tmp_pauli)
        print(tmp_paulis)
        return ''
    
    def is_equal(self,sp):
        out,out_phase=intersection_with_phase([self.paulis,sp.paulis],[self.phases,sp.phases])
        if len(out)==len(self.paulis) and len(out)==len(sp.paulis):
            return True
        else:
            return False
        
    def to_b(self):
        b=[]
        n=self.n
        for ii in range(4**n):
            tmp=self.trace_pauli(ii)
            b.append(tmp)
        b=np.array(b)
        return b





class spd:
    def __init__(self, As, cs):
        self.As=As
        self.cs=cs
        self.n=As[0].n

    def apply_clifford(self, cliff):
        for ii in range(len(self.As)):
            self.As[ii].apply_clifford(cliff)

    def trace_pauli(self, pauli):
        tmp=0
        for ii in range(len(self.As)):
            tmp+=self.cs[ii]*(self.As[ii].trace_pauli(pauli))
        return tmp
    
    def is_commute(self, pauli):
        for ii in range(len(self.As)):
            if not self.As[ii].is_commute(pauli):
                return False
        return True
    
    def find_commute(self, pauli):
        idx=[]
        anti_idx=[]
        for ii in range(len(self.As)):
            if not self.As[ii].is_commute(pauli):
                anti_idx.append(ii)
            else:
                idx.append(ii)
        return idx,anti_idx

    def copy(self):
        As_copy=[]
        for ii in range(len(self.As)):
            As_copy.append(self.As[ii].copy())
        cs_copy=self.cs.copy()
        out=spd(As_copy,cs_copy)
        return out
    
    def baoli_truncate(self,truncate_n):
        for ii in range(len(self.As)):
            self.As[ii].baoli_truncate(truncate_n)
        self.n=self.As[0].n
    
    def tensor_stab_proj(self,A):
        for ii in range(len(self.As)):
            self.As[ii].tensor_stab_proj(A)
        self.n=self.As[0].n
        
    def pad_identity(self,current_idx,n):
        for ii in range(len(self.As)):
            self.As[ii].pad_identity(current_idx,n)
        self.n=self.As[0].n
    
    def __str__(self):
        for ii in range(len(self.As)):
            print('c:',self.cs[ii])
            print('A:',self.As[ii])
        return ''
    
    def norm1_c(self):
        return np.sum(np.abs(self.cs))
    
    def trace(self):
        tmp=0
        for ii in range(len(self.As)):
            tmp+=self.cs[ii]*(self.As[ii].rank)
        return tmp
    
    def norm1_c_w(self):
        tmp=0
        for ii in range(len(self.cs)):
            tmp+=abs(self.cs[ii])*self.As[ii].rank
        return tmp
    
    def extract_from_idx(self,idx):
        As=[self.As[ii] for ii in idx]
        cs=[self.cs[ii] for ii in idx]
        tmp=spd(As,cs)
        return tmp





def channel_decompose(P_U,theta,n):
    #clifford decomposition for exp(-i theta/2 P_U)
    assert -np.pi<=theta and theta<=np.pi

    # 0 <= theta <= pi
    P_U_phase=0
    if theta<=0:
        theta=-theta
        P_U_phase=1

    V=gen_cliff([P_U],[1],[P_U_phase],[0],n)
    #coefficients of I,Z,S
    coes=[(1+np.cos(theta)-np.sin(theta))/2,(1-np.cos(theta)-np.sin(theta))/2,np.sin(theta)]
    cliffds=[]
    #I
    qc=qiskit.QuantumCircuit(n)
    tmp=qiskit.quantum_info.Clifford(qc)
    cliffds.append(tmp)
    #Z
    qc=qiskit.QuantumCircuit(n)
    qc.z(0)
    tmp=qiskit.quantum_info.Clifford(qc)
    tmp=qiskit.quantum_info.Clifford.compose(V,tmp)
    tmp=qiskit.quantum_info.Clifford.compose(tmp,V.adjoint())
    cliffds.append(tmp)
    #S
    qc=qiskit.QuantumCircuit(n)
    qc.s(0)
    tmp=qiskit.quantum_info.Clifford(qc)
    tmp=qiskit.quantum_info.Clifford.compose(V,tmp)
    tmp=qiskit.quantum_info.Clifford.compose(tmp,V.adjoint())
    cliffds.append(tmp)
    return cliffds,coes





def merge_spd(As1,cs1,As2,cs2):
    matched1=[0]*len(As1)
    matched2=[0]*len(As2)
    out_As=[]
    out_cs=[]

    for ii in range(len(matched1)):
        if matched1[ii]!=0:
            continue
        matched1[ii]=1
        out_As.append(As1[ii])
        tmp=cs1[ii]
        
        for jj in range(len(matched2)):
            if matched2[jj]!=0:
                continue
            if As1[ii].is_equal(As2[jj]):
                matched2[jj]=1
                tmp+=cs2[jj]
                break
        
        out_cs.append(tmp)

    for ii in range(len(matched2)):
        if matched2[ii]==0:
            out_As.append(As2[ii])
            out_cs.append(cs2[ii])
    
    return out_As,out_cs





def solve_spd(data,n,perturbation=1):
    M,doc=enumerate_sp(n,refresh=config.refresh)
    M=M.T
    x=cp.Variable(M.shape[1])
    weights=M[0,:]
    if perturbation==0:
        prob = cp.Problem(cp.Minimize(cp.norm(x,1)),
                     [M @ x == data])
    else:
        prob = cp.Problem(cp.Minimize(cp.norm(x,1)+config.perturbation_epsilon*cp.norm(cp.multiply(weights,x),1)),
                     [M @ x == data])

    prob.solve()
        
    x=x.value
    As=[]
    cs=[]
    for jj in range(len(x)):
        if abs(x[jj]*weights[jj])>=config.threshold_epsilon:
            tmp_A=stab_proj(n,doc[jj]['basis'].copy(),doc[jj]['phase'].copy())
            tmp_c=x[jj]
            As.append(tmp_A)
            cs.append(tmp_c)
    spd1=spd(As,cs)
    return spd1





def solve_ssd(data,n,perturbation=1):
    M,doc=enumerate_sp(n,refresh=config.refresh)
    M=M.T
    x=cp.Variable(M.shape[1])
    weights=M[0,:]
    if perturbation==0:
        prob = cp.Problem(cp.Minimize(cp.norm(cp.multiply(weights,x),1)),
                    [M @ x == data])
    else:
        prob = cp.Problem(cp.Minimize(cp.norm(cp.multiply(weights,x),1)+config.perturbation_epsilon*cp.norm(x,1)),
                    [M @ x == data])
        
    prob.solve()
   
    x=x.value
    As=[]
    cs=[]
    for jj in range(len(x)):
        if abs(x[jj]*weights[jj])>=config.threshold_epsilon:
            tmp_A=stab_proj(n,doc[jj]['basis'].copy(),doc[jj]['phase'].copy())
            tmp_c=x[jj]
            As.append(tmp_A)
            cs.append(tmp_c)
    spd1=spd(As,cs)
    return spd1





def resparse_spd(spd1,perturbation=1):
    delta = 1e-6
    NUM_RUNS = 5
    
    bs=[]
    for ii in range(len(spd1.As)):
        bs.append(spd1.As[ii].to_b())
    M=np.array(bs).T
    cs=np.array(spd1.cs)
    data=M@cs
    
    x=cp.Variable(M.shape[1])
    W=cp.Parameter(shape=M.shape[1], nonneg=True);
    W.value=np.ones(M.shape[1])/(delta*np.ones(M.shape[1])+np.absolute(cs))
    
    weights=M[0,:]
    if perturbation==0:
        prob = cp.Problem(cp.Minimize(cp.norm(x,1)+config.perturbation_epsilon*(W.T@cp.abs(x))),
                     [M @ x == data])
    else:
        prob = cp.Problem(cp.Minimize(cp.norm(x,1)+config.perturbation_epsilon*cp.norm(cp.multiply(weights,x),1)
                                     +(config.perturbation_epsilon**2)*(W.T@cp.abs(x))),
                     [M @ x == data])
    
    for k in range(NUM_RUNS):
        prob.solve()
        W.value=np.ones(M.shape[1])/(delta*np.ones(M.shape[1])+np.absolute(x.value))
    
    x=x.value
    n=spd1.n
    As=[]
    cs=[]
    for jj in range(len(x)):
        if abs(x[jj]*weights[jj])>=config.threshold_epsilon:
            tmp_A=stab_proj(n,spd1.As[jj].paulis.copy(),spd1.As[jj].phases.copy())
            tmp_c=x[jj]
            As.append(tmp_A)
            cs.append(tmp_c)
    spd2=spd(As,cs)
    return spd2





def resparse_ssd(spd1,perturbation=1):
    delta = 1e-6
    NUM_RUNS = 5
    
    bs=[]
    for ii in range(len(spd1.As)):
        bs.append(spd1.As[ii].to_b())
    M=np.array(bs).T
    cs=np.array(spd1.cs)
    data=M@cs
    
    x=cp.Variable(M.shape[1])
    W=cp.Parameter(shape=M.shape[1], nonneg=True);
    W.value=np.ones(M.shape[1])/(delta*np.ones(M.shape[1])+np.absolute(cs))
    
    weights=M[0,:]
    if perturbation==0:
        prob = cp.Problem(cp.Minimize(cp.norm(cp.multiply(weights,x),1)+config.perturbation_epsilon*W.T@cp.abs(x)),
                    [M @ x == data])
    else:
        prob = cp.Problem(cp.Minimize(cp.norm(cp.multiply(weights,x),1)+config.perturbation_epsilon*cp.norm(x,1)
                                     +(config.perturbation_epsilon**2)*W.T@cp.abs(x)),
                    [M @ x == data])

    for k in range(NUM_RUNS):
        prob.solve()
        W.value=np.ones(M.shape[1])/(delta*np.ones(M.shape[1])+np.absolute(x.value))
    
    x=x.value
    n=spd1.n
    As=[]
    cs=[]
    for jj in range(len(x)):
        if abs(x[jj]*weights[jj])>=config.threshold_epsilon:
            tmp_A=stab_proj(n,spd1.As[jj].paulis.copy(),spd1.As[jj].phases.copy())
            tmp_c=x[jj]
            As.append(tmp_A)
            cs.append(tmp_c)
    spd2=spd(As,cs)
    return spd2





def noncliff2spd_local(spd1,P_U,theta,ssd=0):
    n=spd1.n
    
    if (ssd==0 and n>=5) or (ssd==1 and n>=5):
        decomposed,decomposed_coes=channel_decompose(P_U,theta,n)
        As=[]
        cs=[]
        for jj in range(len(decomposed)):
            tmp_As=[]
            tmp_cs=[]
            for ii in range(len(spd1.As)):
                tmp=spd1.As[ii].copy()
                tmp.apply_clifford(decomposed[jj])
                tmp_As.append(tmp)
                tmp_cs.append(spd1.cs[ii]*decomposed_coes[jj])
            As,cs=merge_spd(As,cs,tmp_As,tmp_cs)
        spd2=spd(As,cs)
        
        if ssd==0:
            spd2=resparse_spd(spd2,perturbation=config.perturbation)
        else:
            spd2=resparse_ssd(spd2,perturbation=config.perturbation)
        return spd2
    
    b=[]
    decomposed,decomposed_coes=channel_decompose(P_U,-theta,n)
    for ii in range(4**n):
        tmp=0
        for jj in range(len(decomposed)):
            tmp_P=pauli_int2str(ii,n)
            tmp_P=(qiskit.quantum_info.Pauli(tmp_P)).evolve(decomposed[jj].adjoint()).to_label()
            tmp_phase=1
            if tmp_P[0]=='-':
                tmp_phase=-1
                tmp_P=tmp_P[1:]
            tmp_P=pauli_str2int(tmp_P)
            tmp+=tmp_phase*decomposed_coes[jj]*spd1.trace_pauli(tmp_P)
        b.append(tmp)
    b=np.array(b)
    if ssd==0:
        spd2=solve_spd(b,n,config.perturbation)
    else:
        spd2=solve_ssd(b,n,config.perturbation)
    
    if ssd==0:
        spd2=resparse_spd(spd2,perturbation=config.perturbation)
    else:
        spd2=resparse_ssd(spd2,perturbation=config.perturbation)
    return spd2





def noncliff2spd(spd1,P_U,theta,ssd=0):
    if spd1.is_commute(P_U):
        return spd1

    n=spd1.n
    A=[ii.paulis for ii in spd1.As]
    A_phase=[ii.phases for ii in spd1.As]
    

    if len(A)<=1000:
        V,n1,n2=exploit_loc(A,A_phase,P_U,0,n)
    else:
        n1=0
        n2=0


    if n1+n2>0:
        spd1.apply_clifford(V)
        spd1.baoli_truncate(n1+n2)
    
        P_U=pauli_int2str(P_U,n)
        P_U=(qiskit.quantum_info.Pauli(P_U)).evolve(V.adjoint()).to_label()
        if P_U[0]=='-':
            theta=-theta
            P_U=P_U[1:]
        P_U=pauli_str2int(P_U)
        assert P_U%(4**(n1+n2))==0
        P_U=P_U//(4**(n1+n2))
    
        spd2=noncliff2spd_local(spd1,P_U,theta,ssd)
    
        pad=stab_proj(n1+n2,[4**(ii) for ii in range(n1)],[0]*n1)
        spd2.tensor_stab_proj(pad)
        spd2.apply_clifford(V.adjoint())
    
    else:
        spd2=noncliff2spd_local(spd1,P_U,theta,ssd)

    return spd2





def matrix2spd(matrix,ssd=0):
    n=int(np.log2(matrix.shape[0]))
    b=np.zeros(4**n)
    for ii in range(4**n):
        pauli=qiskit.quantum_info.Pauli(pauli_int2str(ii,n)).to_matrix()
        b[ii]=np.trace(pauli @ matrix)
    if ssd==0:
        spd1=solve_spd(b,n,config.perturbation)
    else:
        spd1=solve_ssd(b,n,config.perturbation)
    
    if ssd==0:
        spd1=resparse_spd(spd1,perturbation=config.perturbation)
    else:
        spd1=resparse_ssd(spd1,perturbation=config.perturbation)

    return spd1





def union_local_idx(idx1,idx2):
    idx1=idx1.copy()
    idx2=idx2.copy()
    idx=list(set().union(idx1,idx2))
    idx=sorted(idx)
    idx_map=dict()
    for ii in range(len(idx)):
        idx_map[idx[ii]]=ii
    return idx_map





def qiskit_align_idx(gate,idx_map,idx):
    qc=qiskit.QuantumCircuit(len(idx_map))
    qc.append(gate,[idx_map[ii] for ii in idx])
    return qc





def spd2matrix(spd):
    n=spd.n
    matrix=np.zeros((2**n,2**n))
    for ii in range(len(spd.As)):
        stab_proj=spd.As[ii]
        stab_proj_matrix=np.eye(2**n)
        for jj in range(len(stab_proj.paulis)):
            pauli=qiskit.quantum_info.Pauli(pauli_int2str(stab_proj.paulis[jj],n))
            pauli=pauli.to_matrix()
            if stab_proj.phases[jj]==1:
                pauli=-pauli
            stab_proj_matrix=stab_proj_matrix@((np.eye(2**n)+pauli)/2)
        matrix=matrix+stab_proj_matrix*spd.cs[ii]
    return matrix





def SPD_gen(qc,position,error):
    n=qc.num_qubits
    
    current_idx=[ii.index for ii in qc[position][1]]
    idx_map=union_local_idx(current_idx,[])
    
    tmp_qc=qiskit_align_idx(qc[position][0],idx_map,current_idx)

    current_idx=sorted(current_idx)
    
    U1=qiskit.quantum_info.Operator(tmp_qc).data
    U2=error
    psi,omega1,success=Udiscrimination(U1,U2)
    rho=np.outer(psi,psi.conjugate())
    M=np.outer(omega1,omega1.conjugate())
    spd_rho=matrix2spd(rho,ssd=1)
    spd_M=matrix2spd(M)
    
    #backward
    print('Backward:')
    for ii in tqdm(range(position-1,-1,-1)):
        u_idx=[jj.index for jj in qc[ii][1]]
        if set(current_idx).intersection(set(u_idx))==set():
            continue
            
        idx_map=union_local_idx(current_idx,u_idx)
        if len(idx_map)>len(current_idx):
            spd_rho.pad_identity([idx_map[jj] for jj in current_idx],len(idx_map))
            current_idx=sorted(list(idx_map.keys()))
        
        #clifford
        if qc[ii][0].name in config.clifford_gates:
            tmp_qc=qiskit_align_idx(qc[ii][0].inverse(),idx_map,u_idx)
            spd_rho.apply_clifford(qiskit.quantum_info.Clifford(tmp_qc))
        
        #non-clifford
        elif qc[ii][0].name in config.non_clifford_gates:
            if qc[ii][0].name=='t':
                P_U=4**(idx_map[u_idx[0]])
                theta=-np.pi/4   # tdg
                spd_rho=noncliff2spd(spd_rho,P_U,theta,ssd=1)
            elif qc[ii][0].name=='tdg':
                P_U=4**(idx_map[u_idx[0]])
                theta=np.pi/4    # t
                spd_rho=noncliff2spd(spd_rho,P_U,theta,ssd=1)
            elif qc[ii][0].name=='rz':
                P_U=4**(idx_map[u_idx[0]])
                theta=-qc[ii][0].params[0]  # rzdg
                spd_rho=noncliff2spd(spd_rho,P_U,theta,ssd=1)
            elif qc[ii][0].name=='ry':
                P_U=4**(idx_map[u_idx[0]])+2*(4**(idx_map[u_idx[0]]))
                theta=-qc[ii][0].params[0]  # rydg
                spd_rho=noncliff2spd(spd_rho,P_U,theta,ssd=1)
        else:
            raise ValueError('unknown gate')

    if len(current_idx)<=n:
        spd_rho.pad_identity(current_idx,n)
    rho_trace=spd_rho.trace()
    for ii in range(len(spd_rho.cs)):
        spd_rho.cs[ii]/=rho_trace
    
    current_idx=[ii.index for ii in qc[position][1]]
    idx_map=union_local_idx(current_idx,[])
    current_idx=sorted(current_idx)

    #forward
    print('Forward:')
    for ii in tqdm(range(position+1,len(qc),1)):
        u_idx=[jj.index for jj in qc[ii][1]]
        if set(current_idx).intersection(set(u_idx))==set():
            continue
    
        idx_map=union_local_idx(current_idx,u_idx)
        if len(idx_map)>len(current_idx):
            spd_M.pad_identity([idx_map[jj] for jj in current_idx],len(idx_map))
            current_idx=sorted(list(idx_map.keys()))
        
        #clifford
        if qc[ii][0].name in config.clifford_gates:
            tmp_qc=qiskit_align_idx(qc[ii][0],idx_map,u_idx)
            spd_M.apply_clifford(qiskit.quantum_info.Clifford(tmp_qc))
        
        #non-clifford
        elif qc[ii][0].name in config.non_clifford_gates:
            if qc[ii][0].name=='t':
                P_U=4**(idx_map[u_idx[0]])
                theta=np.pi/4
                spd_M=noncliff2spd(spd_M,P_U,theta)
            elif qc[ii][0].name=='tdg':
                P_U=4**(idx_map[u_idx[0]])
                theta=-np.pi/4
                spd_M=noncliff2spd(spd_M,P_U,theta)
            elif qc[ii][0].name=='rz':
                P_U=4**(idx_map[u_idx[0]])
                theta=qc[ii][0].params[0]
                spd_M=noncliff2spd(spd_M,P_U,theta)
            elif qc[ii][0].name=='ry':
                P_U=4**(idx_map[u_idx[0]])+2*(4**(idx_map[u_idx[0]]))
                theta=qc[ii][0].params[0]
                spd_M=noncliff2spd(spd_M,P_U,theta)
        else:
            raise ValueError('unknown gate')
    
    if len(current_idx)<=n:
        spd_M.pad_identity(current_idx,n)
    
    return spd_rho,spd_M,success
