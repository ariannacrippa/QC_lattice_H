import qiskit
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

class Ansatz:
    """Collection of Ansaetze for the QED circuit."""

    def __init__(self,l,ngauge,nfermions)->None:

        self.l = l
        self.ngauge = ngauge
        self.nfermions = nfermions

        self.n_qubits =int(np.ceil(np.log2(2 * self.l+ 1)))


        self.gray_code_lim()
        self.puregauge_circuit_entang()
        self.fermionic_circuit()


    def gray_code_lim(self,theta=None,layers=1):
        """Return Gray code circuit for gauge field and truncation l=1,2,3,6,7,15."""
        if self.l not in [1,2,3,6,7,15]:
            raise ValueError('l must be 1,2,3,6,7,15')

        qc = QuantumCircuit(self.n_qubits)
        params = lambda i: Parameter(f'theta_{i}')

        if theta is None:
            t=0
        else:
            t=theta

        #First layer of ry gates
        for j in range(layers):
            for i in range(self.n_qubits-1):
                qc.ry(params(t),i)
                t+=1

        if self.l==2:
            for j in range(layers):
                qc.x(0)
                qc.mcry(params(t),list(range(self.n_qubits-1)),self.n_qubits-1,use_basis_gates=True)
                qc.x(0)
                t+=1

        else:
            #1st cry gate
            for j in range(layers):
                qc.cry(params(t),self.n_qubits-2,self.n_qubits-1)
                t+=1

            if self.l==6:
                for j in range(layers):
                    qc.x(0)
                    qc.x(2)
                    qc.mcry(params(t),list(range(self.n_qubits-1)),self.n_qubits-1,use_basis_gates=True)
                    qc.x(0)
                    qc.x(2)
                    t+=1

            #2nd cry gate
            if self.l>1 and self.l!=6:
                for j in range(layers):
                    qc.cry(params(t),self.n_qubits-3,self.n_qubits-1)
                    t+=1

                # if self.l==14:#TODO see why qiskit error
                #     qc.x(0)
                #     qc.x(2)
                #     qc.x(3)
                #     #qc.mcry(params(t),list(range(self.n_qubits-1)),self.n_qubits-1,use_basis_gates=True)
                #     qc.x(0)
                #     qc.x(2)
                #     qc.x(3)
                #     t+=1

                #layers of cry gates if self.l>3
                if self.l>3 and self.l!=14:
                    for j in range(layers):
                        for i in range(1,self.n_qubits-2)[::-1]:
                            qc.cry(params(t),i-1,i)
                            t+=1

        par_list = [params(i) for i in range(t)]

        # self.qc=qc
        # self.par_list=par_list

        return qc,par_list

    # def puregauge_circuit(self):
    #     """Return circuit for pure gauge case, i.e. no fermions.
    #     It considers n gauge field and entangle every nth field with previous ones."""

    #     qgaug = QuantumRegister(self.n_qubits*self.ngauge,name='gaug')
    #     qc_gauge = QuantumCircuit(qgaug)

    #     #list of thetas that will be used to start from vacuum i.e. ..010101
    #     first_layer_par = [0,]
    #     th_gauge=0
    #     #first gauge field
    #     qc_gauge.compose(self.qc,list(range(self.n_qubits)),inplace=True)
    #     # qc_gauge.barrier()
    #     th_gauge=int(''.join(list(filter(str.isdigit, str(self.par_list[-1])))))+1

    #     for i in range(1,self.ngauge):
    #         qc_gauge.ry(Parameter(f'theta_{th_gauge}'),self.n_qubits*i)
    #         first_layer_par+=[th_gauge,]
    #         th_gauge+=1
    #         for j in range(self.n_qubits*i):
    #             qc_gauge.cry(Parameter(f'theta_{th_gauge}'),j,self.n_qubits*i)
    #             th_gauge+=1
    #         # qc_gauge.barrier()
    #         qc_gauge.cry(Parameter(f'theta_{th_gauge}'),self.n_qubits*i,self.n_qubits*i+1)
    #         th_gauge+=1
    #         # qc_gauge.barrier()
    #         #multi-controlled gates
    #         for j in range(self.n_qubits*i):
    #             qc_gauge.mcry(Parameter(f'theta_{th_gauge}'),[j,self.n_qubits*i],self.n_qubits*i+1)
    #             th_gauge+=1

    #     self.first_layer_par=first_layer_par
    #     self.qc_gauge=qc_gauge
    #     self.th_gauge=th_gauge

    def puregauge_circuit_entang(self,entanglement='linear',rzlayer=False,nlayers=1):
        """Return circuit of n gauge fields with Gray encoding and no fermions.
        Entanglement structure between gauge fields for l=1,3,7
         with options: linear, full, none """

        #full entanglement or linear
        if entanglement=='linear':
            cry_gates =lambda i: range(0,self.n_qubits*i,self.n_qubits)
            mcry_gates = lambda i: range(self.n_qubits*i-1,self.n_qubits*i,self.n_qubits)
            mcry7_gates = lambda i: range(0,self.n_qubits*i,self.n_qubits)
        elif entanglement=='full':
            cry_gates =lambda i: range(self.n_qubits*i)
            mcry_gates = lambda i: range(self.n_qubits*i)
            mcry7_gates = lambda i: range(self.n_qubits*i)
        elif entanglement=='triangular':
            if self.ngauge<=2:
                raise ValueError('Triangular entanglement implemented for n gauge fields>2.')
            #layer of cry triangular
            ctr_gates_ry1 = list(range(0,self.n_qubits*((self.ngauge//2)-(self.ngauge+1)%2),self.n_qubits))
            ctr_gates_ry2 = list(range(self.n_qubits*((self.ngauge+(self.ngauge)%2)//2),self.n_qubits*self.ngauge,self.n_qubits))[::-1]

            mctr_gates_ry1 = list(range(self.n_qubits-1,self.n_qubits*((self.ngauge//2)-(self.ngauge+1)%2),self.n_qubits))
            mctr_gates_ry2 = list(range(self.n_qubits*self.ngauge-1,self.n_qubits*((self.ngauge+(self.ngauge)%2)//2),-self.n_qubits))

        elif 'none':
            cry_gates =lambda i: range(0)
        else:
            raise ValueError('Invalid entanglement.')

        qgaug = QuantumRegister(self.n_qubits*self.ngauge,name='gaug')
        qc_gauge = QuantumCircuit(qgaug)

        th_gauge=0
        #first gauge field
        qc_gauge.compose(self.gray_code_lim(theta=th_gauge,layers=nlayers)[0],list(range(self.n_qubits)),inplace=True)
        qc_gauge.barrier()
        th_gauge=int(''.join(list(filter(str.isdigit, str(self.gray_code_lim(theta=th_gauge,layers=nlayers)[1][-1])))))+1
        first_layer_par = [self.n_qubits-2,]

        if entanglement=='triangular':
            qc_gauge.compose(self.gray_code_lim(theta=th_gauge,layers=nlayers)[0],list(range(self.n_qubits*(self.ngauge-1),self.n_qubits*self.ngauge)),inplace=True)

            first_layer_par+=[th_gauge,]

            th_gauge=int(''.join(list(filter(str.isdigit, str(self.gray_code_lim(theta=th_gauge,layers=nlayers)[1][-1])))))+1



            for i in range(1,self.ngauge-1):
                for k in range(nlayers):#first gate of gray structure
                    for n in range(self.n_qubits*i,self.n_qubits*(i+1)-1):
                        qc_gauge.ry(Parameter(f'theta_{th_gauge}'),n)
                        if n==self.n_qubits*(i+1)-2:
                            first_layer_par+=[th_gauge,]
                        th_gauge+=1


            for k in range(nlayers):
                for n in range(self.n_qubits-1):
                    for ctrl in ctr_gates_ry1:
                        qc_gauge.cry(Parameter(f'theta_{th_gauge}'),ctrl+n,ctrl+n+self.n_qubits)
                        th_gauge+=1
                    for ctrl in ctr_gates_ry2:
                        qc_gauge.cry(Parameter(f'theta_{th_gauge}'),ctrl+n,ctrl+n-self.n_qubits)
                        th_gauge+=1



            for i in range(1,self.ngauge-1):
                for k in range(nlayers):#second gate of gray structure for every l=1,3,7
                    qc_gauge.cry(Parameter(f'theta_{th_gauge}'),self.n_qubits*(i+1)-2,self.n_qubits*(i+1)-1)
                    th_gauge+=1



            #multi-controlled gates
            for k in range(nlayers):
                for j in mctr_gates_ry1:

                    qc_gauge.mcry(Parameter(f'theta_{th_gauge}'),[j,j+self.n_qubits-1],j+self.n_qubits)
                    th_gauge+=1


            for k in range(nlayers):
                for j in mctr_gates_ry2:

                    qc_gauge.mcry(Parameter(f'theta_{th_gauge}'),[j,j-self.n_qubits-1],j-self.n_qubits)
                    th_gauge+=1
                    qc_gauge.barrier()

            for i in range(1,self.ngauge-1):

                if self.l==3 or self.l==7:
                    for k in range(nlayers):#third gate of gray structure
                        qc_gauge.cry(Parameter(f'theta_{th_gauge}'),self.n_qubits*(i+1)-3,self.n_qubits*(i+1)-1)
                        th_gauge+=1

                if self.l==7:
                    for k in range(nlayers):#third gate of gray structure
                        qc_gauge.cry(Parameter(f'theta_{th_gauge}'),self.n_qubits*i,self.n_qubits*i+1)
                        th_gauge+=1



        else:
            for i in range(1,self.ngauge):
                for k in range(nlayers):#first gate of gray structure
                    for n in range(self.n_qubits*i,self.n_qubits*(i+1)-1):
                        qc_gauge.ry(Parameter(f'theta_{th_gauge}'),n)
                        if n==self.n_qubits*(i+1)-2:
                            first_layer_par+=[th_gauge,]
                        th_gauge+=1
                for k in range(nlayers):
                    for n in range(self.n_qubits-1):
                        for j in cry_gates(i):

                            qc_gauge.cry(Parameter(f'theta_{th_gauge}'),j+n,self.n_qubits*i+n)
                            th_gauge+=1
                    #qc_gauge.barrier()

                for k in range(nlayers):#second gate of gray structure for every l=1,3,7
                    qc_gauge.cry(Parameter(f'theta_{th_gauge}'),self.n_qubits*(i+1)-2,self.n_qubits*(i+1)-1)
                    th_gauge+=1

                #multi-controlled gates
                for k in range(nlayers):
                    for j in mcry_gates(i):
                        qc_gauge.mcry(Parameter(f'theta_{th_gauge}'),[j,self.n_qubits*(i+1)-2],self.n_qubits*(i+1)-1)
                        th_gauge+=1
                        #qc_gauge.barrier()


                if self.l==3 or self.l==7:
                    for k in range(nlayers):#third gate of gray structure
                        qc_gauge.cry(Parameter(f'theta_{th_gauge}'),self.n_qubits*(i+1)-3,self.n_qubits*(i+1)-1)
                        th_gauge+=1
                        #qc_gauge.barrier()

                    # #multi-controlled gates
                    # for k in range(nlayers):
                    #     for j in mcry_gates(i):
                    #         qc_gauge.mcry(Parameter(f'theta_{th_gauge}'),[j,self.n_qubits*(i+1)-3],self.n_qubits*(i+1)-1)
                    #         th_gauge+=1
                    #         #qc_gauge.barrier()

                if self.l==7:
                    for k in range(nlayers):#third gate of gray structure
                        qc_gauge.cry(Parameter(f'theta_{th_gauge}'),self.n_qubits*i,self.n_qubits*i+1)
                        th_gauge+=1
                        #qc_gauge.barrier()

                    # #multi-controlled gates
                    # for k in range(nlayers):
                    #     for j in mcry7_gates(i):
                    #         qc_gauge.mcry(Parameter(f'theta_{th_gauge}'),[j,self.n_qubits*i],self.n_qubits*i+1)
                    #         th_gauge+=1
                    #         #qc_gauge.barrier()

        if rzlayer:
            for k in range(self.n_qubits*self.ngauge):
                qc_gauge.rz(Parameter(f'theta_{th_gauge}'), k)
                th_gauge+=1

        return qc_gauge,first_layer_par,th_gauge
        # self.qc_gauge=qc_gauge
        # self.first_layer_par=first_layer_par
        # self.th_gauge=th_gauge

    def fermionic_circuit(self):
        """Return circuit for fermionic case, i.e. no gauge fields.
           It considers iSwap gates between every two fermions in order to select only zero-charged states."""

        qferm = QuantumRegister(self.nfermions,name='ferm')
        qc_ferm = QuantumCircuit(qferm)

        th_ferm=0
        #fermion circuit proposal
        p = Parameter('p')
        qc_iswap = QuantumCircuit(2,name='iSWAP')
        qc_iswap.rxx(p, 0, 1)
        qc_iswap.ryy(p, 0, 1)

        params = lambda i: Parameter(f'theta_{i}')

        for i in range(0,self.nfermions,2):
            qc_ferm.x(qferm[i])

        for j in range(self.nfermions//2):
            for i in range(j,self.nfermions-j,2):
                    qc_ferm.append(qc_iswap.to_instruction({p: params(th_ferm)}), qferm[i:i+2])
                    th_ferm+=1

        #last layer of Rz gates for correct phase
        for i in range(self.nfermions):
            qc_ferm.rz(Parameter(f'theta_{th_ferm}'),qferm[i])
            th_ferm+=1

        self.qc_ferm=qc_ferm
        self.th_ferm=th_ferm


        #TODO add gauge+fermionic circuit



    #Rule to see how many parameters I need
    def parameters_count(self,n_qubits,ngauge):#TODO add fermions
        return n_qubits*ngauge+n_qubits*sum(i for i in range(2,n_qubits*(ngauge-1)+1,2))+n_qubits*ngauge