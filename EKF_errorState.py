import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from quat_helper import *
from scipy.spatial.transform import Rotation

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

#
class Quaternion_EKF:
    def __init__(self):
        
        omegaUpdate = True
        simultaneousUpdate = True
        self.q = np.array([1,0,0,0]).reshape(-1,1)
        self.Q_theta = 0.08*np.eye(3)
        self.Q_omega = 0.008*np.eye(3)
        self.Q = np.eye(6)
        self.Q[:3,:3] = self.Q_theta
        self.Q[3:,3:] = self.Q_omega

        self.P = 10.0 * np.eye(self.Q.shape[0])
        
        # self.R_omega = 0.001*np.eye(3)
        if omegaUpdate:
            self.R_acc = 0.05*np.eye(3)
        else:
            self.R_acc = 0.05*np.eye(4)
        if simultaneousUpdate:
            self.R_acc = 0.05*np.eye(6)
        self.R_q = 0.1*np.eye(4)
        self.X = np.zeros((self.Q.shape[0],1))
        self.StateE = np.zeros((self.Q.shape[0],1))
        self.count = 0
        self.omega_hat_ = np.zeros((3,1))
        self.gravity = np.array([0,0,-1]).reshape(-1,1)
        self.magneto = np.array([1,0,0]).reshape(-1,1)
        self.Omega_b = np.zeros((3,1))
    
    def skewSymMat(self,vec):
        vec=vec.reshape(-1,1)
        return np.array([[0,-vec[2,0],vec[1,0]],\
                         [vec[2,0],0,-vec[0,0]],\
                            [-vec[1,0],vec[0,0],0]])

    def phi(self,delq):#map deltaQ to delta Theta
        return 2*delq[1:]

    def q_L(self,q):
        qw,qx,qy,qz = q.ravel()[:]
        return np.array([[qw, -qx, -qy, -qz],\
                            [qx, qw, -qz, qy],\
                            [qy, qz, qw, -qx],\
                            [qz, -qy, qx, qw]])

    def Quaternion(self,del_theta):
        theta = np.linalg.norm(del_theta)
        dir = del_theta/theta
        if theta ==0:
            return np.append([1],np.zeros(3)).reshape(-1,1)
        else: 
            return np.append([np.cos(theta/2)],dir*np.sin(theta/2)).reshape(-1,1)

    def quaternion_congu(self,q):
        q=q.reshape(-1,1)
        q[1:] *= -1
        return q

    def Predict(self,omega_m,dt):
        # print (self.Omega_b.ravel(),omega_m.ravel())
        self.F = np.append(np.append(self.skewSymMat(dt*(omega_m.ravel()-self.Omega_b.ravel())).T,-np.eye(3)*dt,axis=1),np.append(np.zeros((3,3)),np.eye(3),axis=1),axis=0)
        # delta_x = np.matmul(self.F,delta_x) = 0# all error states are bought to zero
        self.P = np.matmul(self.F,np.matmul(self.P,self.F.T)) + self.Q

    def update2(self, meas_vec):
        meas_vec = meas_vec.reshape(4,1)# this is quaternion measurement

        # self.Predict(meas_omega,dt)
        
        _v = -self.q[1:,0].reshape(3,1)
        _w = self.q[0,0]
        # dau_h_by_dau_q = 2*np.append([_w*meas_vec+np.matmul(self.skewSymMat(_v),meas_vec)],\
        #                             np.matmul(_v.T,meas_vec)*np.eye(3)+np.matmul(_v,meas_vec.T)-np.matmul(meas_vec,_v.T)-_w*self.skewSymMat(meas_vec),axis=0)
        
        # dau_h_by_dau_q = 2*np.append(_w*self.gravity + self.skewSymMat(_v)@self.gravity,\
        #                             (_v.T@self.gravity)[0,0]*np.eye(3) + _v@self.gravity.T-self.gravity@_v.T-_w*self.skewSymMat(self.gravity),axis=1)
        
        dau_q_by_dau_del_theta = 0.5*self.q_L(self.q)@np.append([[0,0,0]],np.eye(3),axis=0)
        
        self.H = np.append(dau_q_by_dau_del_theta,np.zeros((4,3)),axis=1)
        
        K = np.matmul(self.P,np.matmul(self.H.T, np.linalg.inv(np.matmul(self.H,np.matmul(self.P,self.H.T))+self.R_q)))
        # print (meas_vec,- rotate_vector_by_quaternion(self.quaternion_congu(self.q.copy()), meas_vec))
        self.delta_x = np.matmul(K, (meas_vec - self.q.copy()))
        # self.delta_x = np.matmul(K, (rotate_vector_by_quaternion(self.q.copy(),meas_vec) - self.gravity))
        # print (meas_vec,- rotate_vector_by_quaternion(self.quaternion_congu(self.q), meas_vec))
        self.P = np.matmul((np.eye(6)-np.matmul(K,self.H)),np.matmul(self.P,(np.eye(6)-np.matmul(K,self.H)).T)) + np.matmul(K,np.matmul(self.R_q,K.T))

        self.q = multiply_quaternions(self.q, self.Quaternion(self.delta_x[:3,0]))
        self.Omega_b += self.delta_x[3:,0].reshape(-1,1)

        self.G = np.eye(6)
        self.G[:3,:3] -= 0.5*self.skewSymMat(self.delta_x[:3,0])
        self.P = self.G@self.P@self.G.T
        return self.q

    def Update(self, meas_vec, meas=True):
        meas_vec = meas_vec.reshape(-1,1)# this is gravity measured by the sensor in body frame>> g' =  R{q(t)}^T * g
    
        # self.Predict(meas_omega,dt)
        
        _v = self.q[1:,0].reshape(3,1)
        _w = self.q[0,0]
        # dau_h_by_dau_q = 2*np.append([_w*meas_vec+np.matmul(self.skewSymMat(_v),meas_vec)],\
        #                             np.matmul(_v.T,meas_vec)*np.eye(3)+np.matmul(_v,meas_vec.T)-np.matmul(meas_vec,_v.T)-_w*self.skewSymMat(meas_vec),axis=0)
        dau_q_by_dau_del_theta = 0.5*self.q_L(self.q)@np.append([[0,0,0]],np.eye(3),axis=0)
        if meas:
            self.R_acc = 0.05*np.eye(3)

            dau_h_by_dau_q = 2*np.append(_w*self.gravity + self.skewSymMat(self.gravity)@_v,\
                                        (_v.T@self.gravity)[0,0]*np.eye(3) + _v@self.gravity.T-self.gravity@_v.T+_w*self.skewSymMat(self.gravity),axis=1)
            self.H = np.append(dau_h_by_dau_q@dau_q_by_dau_del_theta, np.zeros((3,3)),axis= 1)

        else:
            self.R_acc = 0.05*np.eye(6)

            dau_h_by_dau_q_grav = 2*np.append(_w*self.gravity + self.skewSymMat(self.gravity)@_v,\
                                        (_v.T@self.gravity)[0,0]*np.eye(3) + _v@self.gravity.T-self.gravity@_v.T+_w*self.skewSymMat(self.gravity),axis=1)
            dau_h_by_dau_q_grav = np.append(dau_h_by_dau_q_grav@dau_q_by_dau_del_theta,np.zeros((3,3)),axis=1)

            dau_h_by_dau_q_mag = 2*np.append(_w*self.magneto + self.skewSymMat(self.magneto)@_v,\
                                        (_v.T@self.magneto)[0,0]*np.eye(3) + _v@self.magneto.T-self.magneto@_v.T+_w*self.skewSymMat(self.magneto),axis=1)
            dau_h_by_dau_q_mag = np.append(dau_h_by_dau_q_mag @ dau_q_by_dau_del_theta,np.zeros((3,3)),axis=1)
            self.H = np.append(dau_h_by_dau_q_mag,dau_h_by_dau_q_grav,axis=0)
        # print ("H:\n",dau_h_by_dau_q,"\n")
        
        
        # dau_q_by_dau_del_theta = np.append(dau_q_by_dau_del_theta,np.zeros((4,3)),axis=1) # dau q/dau delx 
        # dau_q_by_dau_del_theta = np.append(dau_q_by_dau_del_theta,np.append(np.zeros((3,3)),np.eye(3),axis=1),axis=0) # dau x/ dau del x
        # 
        

        
        
        K = np.matmul(self.P,np.matmul(self.H.T, np.linalg.inv(np.matmul(self.H,np.matmul(self.P,self.H.T))+self.R_acc)))
        # print (K)
        # print (meas_vec,- rotate_vector_by_quaternion(self.quaternion_congu(self.q.copy()), meas_vec))
        if meas:
            self.delta_x = np.matmul(K, (meas_vec - rotate_vector_by_quaternion(self.q.copy(), self.gravity)))
        else:
            grav = rotate_vector_by_quaternion(self.q.copy(), self.gravity)
            mag = rotate_vector_by_quaternion(self.q.copy(), self.magneto)
            self.delta_x = np.matmul(K, (meas_vec - np.append(mag,grav,axis=0)))
        print ("delta_x=",self.delta_x.ravel())
        # self.delta_x = np.matmul(K, (rotate_vector_by_quaternion(self.q.copy(),meas_vec) - self.gravity))
        # print (meas_vec,- rotate_vector_by_quaternion(self.quaternion_congu(self.q), meas_vec))
        self.P = np.matmul((np.eye(6)-np.matmul(K,self.H)),np.matmul(self.P,(np.eye(6)-np.matmul(K,self.H)).T)) + np.matmul(K,np.matmul(self.R_acc,K.T))
        
        self.q = multiply_quaternions(self.q, self.Quaternion(self.delta_x[:3,0]))
        self.Omega_b += self.delta_x[3:,0].reshape(-1,1)

        self.G = np.eye(6)
        self.G[:3,:3] -= 0.5*self.skewSymMat(self.delta_x[:3,0])
        self.P = self.G@self.P@self.G.T
        return self.q.copy()

def to_xyzw(q):
    return np.append(q[1:,0],q[0,0]).reshape(-1,1)

def normalize(m):
    m[2,0]=0
    m/=np.linalg.norm(m)
    return m


if __name__=="__main__":
    gravity = np.array([0,0,-1]).reshape(-1,1)
    mag = np.array([1,0,0]).reshape(-1,1)
    # q = np.array([np.cos(np.pi/100),np.sin(np.pi/100),0,0]).reshape(-1,1)
    q = np.array([1,0,0,0]).reshape(-1,1)
    omega = np.zeros(3)
    dt = 0.005
    G=[]
    Q=[]
    Q_noise=[]
    W = []
    EULER = []
    EULER_Est = []
    plt.figure(1)
    ekf = Quaternion_EKF()
    Q_ekf = []
    
    for i in range(1000):
        omega[1] = np.sin(i*dt*0.1*(2*np.pi*10)) #+ np.random.randn()*0.01
        normOmega = np.linalg.norm(omega)
        if normOmega!=0:
            q = multiply_quaternions(np.append([np.cos(normOmega*dt/2)], omega/normOmega*np.sin(normOmega*dt/2)),q).copy()
        else:
            q = multiply_quaternions(np.append([1], np.zeros(3)),q).copy()
        g = rotate_vector_by_quaternion(q.copy(),gravity).copy()
        m = normalize(rotate_vector_by_quaternion(q.copy(),mag).copy())
        print ("magnometer: " ,m.ravel())
        # print ("gravity",g[0,0],g[1,0],g[2,0])
        ekf.Predict(omega,dt)
        # if i%100==0:
        # #     _=ekf.Update(np.append(m,g,axis=0),meas=False)
        #     Q_ekf.append(ekf.Update(np.append(m,g,axis=0),meas=False))
        # else:
        #     Q_ekf.append(ekf.Update(g).copy())
        # Q_ekf.append(ekf.Update(m,meas=False).copy())
        Q.append(q)
        noiseQuaternion = q + np.random.randn(*(q.shape))*0.05
        noiseQuaternion /= np.linalg.norm(noiseQuaternion)
        Q_ekf.append(ekf.update2(noiseQuaternion))
        Q_noise.append(noiseQuaternion)
        euler = Rotation.from_quat(to_xyzw(q.copy()).ravel()).as_euler('xyz', degrees=True).ravel()
        euler__ = Rotation.from_quat(to_xyzw(Q_ekf[-1]).ravel()).as_euler('xyz', degrees=True).ravel()
        EULER.append(euler)
        EULER_Est.append(euler__)
        G.append(g)
        W.append(omega)

    Q = np.array(Q).reshape(-1,4)
    G = np.array(G).reshape(-1,3)
    W = np.array(W).reshape(-1,3)
    EULER = np.array(EULER).reshape(-1,3)
    EULER_Est = np.array(EULER_Est).reshape(-1,3)
    # np.savetxt("omega",W)
    # np.savetxt("gravity",G)

    Q_ekf = np.array(Q_ekf).reshape(-1,4)
    Q_ekf_norm = np.linalg.norm(Q_ekf,axis=1)
    plt.plot(Q_ekf_norm)
    plt.ylim([0,1.5])
    _,ax = plt.subplots(4,1,sharex=True)
    ax[0].plot(Q[:,0],c="k")
    ax[0].plot(Q_ekf[:,0],c="k",ls="--")

    ax[1].plot(Q[:,1],c="k")
    ax[1].plot(Q_ekf[:,1],c="k",ls="--")

    ax[2].plot(Q[:,2],c="k")
    ax[2].plot(Q_ekf[:,2],c="k",ls="--")

    ax[3].plot(Q[:,3],c="k")
    ax[3].plot(Q_ekf[:,3],c="k",ls="--")

    _,ax2 = plt.subplots(3,1,sharex=True)
    ax2[0].plot(EULER[:,0],c="k")
    ax2[0].plot(EULER_Est[:,0],c="k",ls="--")

    ax2[1].plot(EULER[:,1],c="k")
    ax2[1].plot(EULER_Est[:,1],c="k",ls="--")

    ax2[2].plot(EULER[:,2],c="k")
    ax2[2].plot(EULER_Est[:,2],c="k",ls="--")
    
    plt.show()
    # G = np.loadtxt("gravity")
    # W = np.loadtxt("omega")

    # ekf = Quaternion_EKF()
    # Q=[]
    # try:
    #     for i in range(G.shape[0]):
    #         if i ==0:
    #             ekf.omega  = W[i,:]
    #         ekf.Update(G[i,:],W[i,:],dt)
    #         q = ekf.q_n
    #         Q.append(q)
    # except:
    #     print ("failed again")
    # Q = np.array(Q).reshape(-1,4)
    # plt.plot(Q[:,0],c="b")
    # plt.plot(Q[:,1],c="b")
    # plt.plot(Q[:,2],c="b")
    # plt.plot(Q[:,3],c="b")
    # plt.show()
