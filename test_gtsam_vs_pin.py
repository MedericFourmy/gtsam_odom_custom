import numpy as np
import gtsam
import pinocchio as pin

def convert_se3(nu):
    new_nu = np.zeros(6)
    new_nu[:3] = nu[3:]
    new_nu[3:] = nu[:3]
    return new_nu

def convert_jac_se3(J):
    J_new = np.zeros((6,6))
    J_new[:3,:3] = J[3:,3:] 
    J_new[3:,3:] = J[:3,:3] 
    J_new[3:,:3] = J[:3,3:] 
    J_new[:3,3:] = J[3:,:3] 
    return J_new

def minus_se3(T2, T1):
    # T2 - T1)
    return pin.log6(T1.inverse()*T2).vector

def numdiff(f, X, n, m, exp, minus=np.subtract, eps=1e-6):
    Jac = np.zeros((m,n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        Xd = X * exp(e)
        Jac[:,i] = minus(f(Xd), f(X))/eps
    return Jac



Tp1 = pin.SE3.Random()
Tp2 = pin.SE3.Random()

Tg1, Tg2 = gtsam.Pose3(Tp1.homogeneous), gtsam.Pose3(Tp2.homogeneous)

Tp3 = Tp1 * Tp2
Tg3 = Tg1 * Tg2

assert np.allclose(Tp3.homogeneous, Tg3.matrix())

# Log
vg1 = gtsam.Pose3.Logmap(Tg1)
vg1_bis = gtsam.Pose3.logmap(gtsam.Pose3.Identity(), Tg1)
vp1 = pin.log(Tp1).vector
assert np.allclose(vg1, vg1_bis)
assert np.allclose(vg1, convert_se3(vp1))

# Exp
Tg1_Exp = gtsam.Pose3.Expmap(vg1)
assert np.allclose(Tg1_Exp.matrix(), Tg1.matrix())

# Jlog
Jlog_Tp1 = pin.Jlog6(Tp1)
Jlog_Tg1 = np.eye(6, order='F')  # Logmap only accepts f_continuous Jac array 
gtsam.Pose3.Logmap(Tg1, Jlog_Tg1)
assert np.allclose(Jlog_Tp1, convert_jac_se3(Jlog_Tg1))

# Adjoint
Ap1 = Tp1.action
Ag1 = Tg1.AdjointMap()
assert np.allclose(Ap1, convert_jac_se3(Ag1))
# adjoint of the inverse if the inverse of the adjoint
assert np.allclose(Tg1.inverse().AdjointMap(), np.linalg.inv(Ag1))


def to_posi(T: pin.SE3):
    return T.translation

# d M.t / dM
n, m = 6, 3
J_p_n = numdiff(to_posi, Tp1, n, m, pin.exp)
J_p = np.hstack([Tp1.rotation, np.zeros((3,3))])
assert np.allclose(J_p_n, J_p[:3,:])
