import numpy as np
import math

def quat_average(q,q0):
    q = np.matrix(q)
    qt = q0
    nr, nc = np.shape(q)
    qe = np.matrix(np.zeros([nr, 4]))
    ev = np.matrix(np.zeros([nr, 3]))
    pi = np.pi
    epsilon = 0.0001
    temp = np.zeros([1,4])
    for t in range(1000):
        for i in range(0,nr,1):
            q[i] = normalize_quaternion(q[i])
            qe[i] = multiply_quaternions(q[i],inverse_quaternion(qt))
            qs = qe[i,0]
            qv = qe[i,1:4]
            if np.round(norm_quaternion(qv),8) == 0:
                if np.round(norm_quaternion(qe[i]),8) == 0:
                    ev[i] = np.matrix([0, 0, 0])
                else:
                    ev[i] = np.matrix([0, 0, 0])
            if np.round(norm_quaternion(qv),8) != 0:
                if np.round(norm_quaternion(qe[i]),8) == 0:
                    ev[i] = np.matrix([0, 0, 0])
                else:
                    temp[0,0] = np.log(norm_quaternion(qe[i]))
                    temp[0,1:4] = np.dot((qv/norm_quaternion(qv)),math.acos(qs/norm_quaternion(qe[i])))
                    ev[i] = 2*temp[0,1:4]
                    ev[i] = ((-np.pi + (np.mod((norm_quaternion(ev[i]) + np.pi),(2*np.pi))))/norm_quaternion(ev[i]))*ev[i]
        e = np.transpose(np.mean(ev, 0))
        temp2 = np.array(np.zeros([4,1]))
        temp2[0] = 0
        temp2[1:4] = e/2.0
        temp2+=0.00001*np.ones(temp2.shape)
        qt = multiply_quaternions(exp_quaternion(np.transpose(temp2)),qt)

        if norm_quaternion(e) < epsilon:
            return qt, ev

def multiply_quaternions(p, q):
    
    p=p.flatten()
    q=q.flatten()
    # print (np.cross(p[1:,0],q[1:,0]))
    t = np.append([p[0]*q[0]-np.inner(p[1:],q[1:])],p[0]*q[1:]+q[0]*p[1:]+np.cross(p[1:],q[1:])).reshape(-1,1)
    return t


def conjugate_quaternion(q):
    q=q.ravel()
    t = np.empty([4])
    t[0] = q[0]
    t[1] = -q[1]
    t[2] = -q[2]
    t[3] = -q[3]
    return t


def divide_quaternions(q, r):
    q=q.ravel()
    r=r.ravel()
    t = np.empty([4, 1])
    t[0] = ((r[0] * q[0]) + (r[1] * q[1]) + (r[2] * q[2]) + (r[3] * q[3])) / ((r[0] ** 2) + (r[1] ** 2) + (r[2] ** 2) + (r[3] ** 2))
    t[1] = (r[0] * q[1] - (r[1] * q[0]) - (r[2] * q[3]) + (r[3] * q[2])) / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)
    t[2] = (r[0] * q[2] + r[1] * q[3] - (r[2] * q[0]) - (r[3] * q[1])) / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)
    t[3] = (r[0] * q[3] - (r[1] * q[2]) + r[2] * q[1] - (r[3] * q[0])) / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)

    return t


def inverse_quaternion(q):
    t = np.empty([4, 1])
    t[0] = q[:,0] / np.power(norm_quaternion(q),2)
    t[1] = -q[:,1] / np.power(norm_quaternion(q),2)
    t[2] = -q[:,2] / np.power(norm_quaternion(q),2)
    t[3] = -q[:,3] / np.power(norm_quaternion(q),2)

    t = np.transpose(t)

    return t


def norm_quaternion(q):
    t = np.sqrt(np.sum(np.power(q,2)))
    return t


def normalize_quaternion(q):
    return q/norm_quaternion(q)

def rotate_vector_by_quaternion(q,v):
    q=q.ravel()
    v = v.reshape(-1,1)
    v_rotated = []
    v_rotated = np.matmul(np.matrix([[(1 - 2*(q[2]**2) - 2*(q[3]**2)), 2*(q[1]*q[2] + q[0]*q[3]), 2*((q[1]*q[3]) - (q[0]*q[2]))],
                           [2*(q[1]*q[2] - q[0]*q[3]), (1 - 2*(q[1]**2) - 2*(q[3]**2)), 2*((q[2]*q[3]) + (q[0]*q[1]))],
                           [2*(q[1]*q[3] + q[0]*q[2]), 2*((q[2]*q[3]) - (q[0]*q[1])), (1 - 2*(q[1]**2) - 2*(q[2]**2))]]),v)
    return v_rotated

def crossMat(vec):
    vec=vec.ravel()
    return np.array([[0,-vec[2],vec[1]],[vec[2],0,-vec[0]],[-vec[1],vec[0],0]])

def quat2rot(q):
    q = normalize_quaternion(q)
    qhat = crossMat(q[1:,0])

    R = (q[0,0]**2 - np.inner(q[1:,0],q[1:,0]))*np.eye(3) + 2*np.outer(q[1:,0],q[1:,0]) + 2*q[0,0]*qhat
    return R


def rot2euler(R):

    phi = -math.asin(R[1,2])
    theta = -math.atan2(-R[0,2]/math.cos(phi),R[2,2]/math.cos(phi))
    psi = -math.atan2(-R[1,0]/math.cos(phi),R[1,1]/math.cos(phi))

    return phi, theta, psi


def rot2quat(R):

    tr = R[0,0] + R[1,1] + R[2,2];

    if tr > 0:
        S = np.sqrt(tr+1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S

    elif ((R[0,0] > R[1,1]) and (R[0,0] > R[2,2])):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S

    elif (R[1,1] > R[2,2]):
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S

    q = [[qw],[qx],[qy],[qz]]
    temp = np.sign(qw)
    q = np.multiply(q,temp)
    return q


def vec2quat(r):
    r = r.reshape(3,1)
    r = r/2.0
    mag = np.linalg.norm(r)
    
    if mag == 0:
        q = np.append([math.cos(mag)], np.zeros(3)).reshape(-1,1)
    else:
        q = np.append([[math.cos(mag)]], (r/mag)*(math.sin(mag)),axis=0)
    return q


def quat2vec(q):

    qs = q[:,0]
    qv = q[:,1:4]
    if np.linalg.norm(qv) == 0:
        v = np.transpose(np.matrix([0,0,0]))
    else:
        v = 2*((qv/np.linalg.norm(qv))*math.acos(qs/np.linalg.norm(q)))
    return v


def log_quaternion(qe):

    qe = np.transpose(qe)
    qs = qe[0]
    qv = qe[1:4]
    log_q = np.zeros(np.shape(qe))

    log_q[0] = np.log(norm_quaternion(qe))
    log_q[1:4] = np.dot(qv/norm_quaternion(qv), math.acos(qs/norm_quaternion(qe)))
    return log_q


def exp_quaternion(q):

    q = np.transpose(q)
    qs = q[0]
    qv = q[1:4]
    exp_q = np.zeros(np.shape(q))

    exp_q[0] = math.cos((norm_quaternion(qv)))
    exp_q[1:4] = np.dot(normalize_quaternion(qv), math.sin(norm_quaternion(qv)))
    return np.transpose(math.exp(qs)*exp_q)
