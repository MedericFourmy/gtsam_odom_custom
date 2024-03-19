
import numpy as np
import gtsam
import pinocchio as pin


def evl(T1: gtsam.Pose3, T2: gtsam.Pose3, nu12: np.ndarray, dt):
    """
    Encodes relationship: T2 = T1 * Exp(nu12*dt)
    as the error:
    e = Log(T1.inv() * T2) - nu12*dt
    """

    # Compute error
    T1_inv = T1.inverse()
    M12 = T1_inv * T2
    error = gtsam.Pose3.Logmap(M12) - nu12*dt
    return error

def jacs_evl(T1, T2, nu12, dt):
    """
    Jacobian error local velocity
    """
    jacobians = [np.zeros(6) for _ in range(3)]
    
    T1_inv = T1.inverse()
    M12 = T1_inv * T2
    JlogM12 = np.eye(6, order='F')  # Logmap only accepts f_continuous Jac array 
    gtsam.Pose3.Logmap(M12, JlogM12)
    jacobians[0] = JlogM12 @ T2.inverse().AdjointMap() @ (-T1.AdjointMap())
    jacobians[1] = JlogM12
    jacobians[2] = -np.eye(6)*dt

    return jacobians


def evg(T1: gtsam.Pose3, T2: gtsam.Pose3, nu12: np.ndarray, dt):
    """
    Error global velocity

    Encodes relationship: T2 = Exp(nu12*dt) * T1 
    as the error:
    e = Log(T2 * T1.inv()) - nu12*dt
    """

    T1_inv = T1.inverse()
    M12 = T2 * T1_inv 
    error = gtsam.Pose3.Logmap(M12) - nu12*dt
    return error


def jacs_evg(T1, T2, nu12, dt):
    """
    Jacobian error global velocity
    """
    jacobians = [np.zeros((6,6)) for _ in range(3)]
    
    # Compute error
    T1_inv = T1.inverse()
    M12 = T2 * T1_inv 
    # and jacobians
    JlogM12 = np.eye(6, order='F')  # Logmap only accepts f_continuous Jac array 
    gtsam.Pose3.Logmap(M12, JlogM12)
    jacobians[0] = JlogM12 @ (-T1.AdjointMap())
    jacobians[1] = JlogM12 @ T1.AdjointMap()
    jacobians[2] = -np.eye(6)*dt

    return jacobians


def ev_r3so3_g(T1: gtsam.Pose3, T2: gtsam.Pose3, nu12: np.ndarray, dt):
    R1, R2 = T1.rotation(), T2.rotation()
    p1, p2 = T1.translation(), T2.translation()
    omg12, v12 = nu12[:3], nu12[3:]

    # Compute error
    R12 = R2 * R1.inverse() 
    error_o = pin.log3(R12.matrix()) - omg12*dt
    error_p = p2 - p1 - v12*dt
    return np.concatenate([error_o, error_p])


def jacs_ev_r3so3_g(T1, T2, nu12, dt):
    jacobians = [np.zeros((6,6)) for _ in range(3)]
    
    R1, R2 = T1.rotation(), T2.rotation()

    # Compute error
    R12 = R2 * R1.inverse() 
    # TODO: could not find gtsam equivalent
    JlogR12 = pin.Jlog3(R12.matrix())

    jacobians[0][:3,:3] = JlogR12 @ (-R1.matrix())
    jacobians[0][3:6,3:6] = -R1.matrix()
    jacobians[1][:3,:3] = JlogR12 @ R1.matrix()
    jacobians[1][3:6,3:6] = R2.matrix()
    jacobians[2] = -np.eye(6)*dt

    return jacobians


def num_diff(T1, T2, nu12, dt, ev=evl, exp=gtsam.Pose3.Expmap, EPS=1e-6):
    e0 = ev(T1,T2,nu12,dt)
    JT1 = np.zeros((6,6))
    JT2 = np.zeros((6,6))
    Jnu = np.zeros((6,6))
    for i in range(6):
        e = np.zeros(6)
        e[i] = EPS
        JT1[:,i] = (ev(T1*exp(e),T2,nu12,dt) - e0)/EPS
        JT2[:,i] = (ev(T1,T2*exp(e),nu12,dt) - e0)/EPS
        Jnu[:,i] = (ev(T1,T2,nu12+e,dt) - e0)/EPS
    
    return JT1, JT2, Jnu


# Create random poses and twist
nu1, nu2, nu12 = np.random.random(6), np.random.random(6), np.random.random(6)
T1, T2 = gtsam.Pose3.Expmap(nu1), gtsam.Pose3.Expmap(nu2)
dt = 0.2

# Compare analytical with numdiff - LOCAL
JT1_n, JT2_n, Jnu_n = num_diff(T1, T2, nu12, dt, ev=evl)
JT1, JT2, Jnu = jacs_evl(T1, T2, nu12, dt)

assert np.allclose(JT1, JT1_n, atol=1e-7)
assert np.allclose(JT2, JT2_n, atol=1e-7)
assert np.allclose(Jnu, Jnu_n, atol=1e-7)

# Compare analytical with numdiff - GLOBAL
JT1_n, JT2_n, Jnu_n = num_diff(T1, T2, nu12, dt, ev=evg)
JT1, JT2, Jnu = jacs_evg(T1, T2, nu12, dt)

assert np.allclose(JT1, JT1_n, atol=1e-7)
assert np.allclose(JT2, JT2_n, atol=1e-7)
assert np.allclose(Jnu, Jnu_n, atol=1e-7)

# Compare analytical with numdiff - SO3R3 GLOBAL
JT1_n, JT2_n, Jnu_n = num_diff(T1, T2, nu12, dt, ev=ev_r3so3_g)
JT1, JT2, Jnu = jacs_ev_r3so3_g(T1, T2, nu12, dt)

print(JT2)
print(JT2_n)
assert np.allclose(JT1, JT1_n, atol=1e-7)
assert np.allclose(JT2, JT2_n, atol=1e-7)
assert np.allclose(Jnu, Jnu_n, atol=1e-7)
