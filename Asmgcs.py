import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math


#Advanced Surface Movement Guidance and control System

#Surveillance, radar, airfield ground lighting, Vehicle Tracking System, visual docking guidance system

'''
MLAT/WAM, ADS-B Ground station, HMI/CWP, safety logic conflict, airport moving maps, advanced video and auditory warning
'''
altitude = 0
heading = 30
speed = 30
global trajectory
trajectory = []
obstruction_map = [[26, 10], [33, 6], [8, 4]]

from physics import coordinate_mapping
possible_route = []
rowroute = coordinate_mapping()

def sensor(q):
    if q == 0:
        return 20
    else:
        return 30

def gyroscope(t,bearing, acc):
    vx = vy  =  0
    vy = (vy + acc * t) * math.cos(float(bearing))
    vx = (vx + acc * t) * math.sin(float(bearing))
    global y
    global x
    y = vy*t
    x = vx*t
    return [x,y]



def proximity(xp,yp):
    res = True in (((i[0]-xp)**2 + (i[1]-yp)**2)**0.5 <= 10 for i in obstruction_map)
    if res == True:
        print(f'possible collision warning at {track[-1][0], track[-1][1]}')
        print(track[-1])

while True:
    track = []
    i = 0
    while i <= 100:
        b = sensor(0)
        a = sensor(1)
        track.append(gyroscope(i, b, a))
        proximity(track[-1][0], track[-1][1])
        i = i = 0.1
    continue

#Planning

#Routing

# offsets of each variable in the state vector
iX = 0
iV = 1
NUMVARS = iV + 1

#kalman filter
class KF:
    def __init__(self, initial_x: float,
                 initial_v: float,
                 accel_variance: float) -> None:
        # mean of state GRV
        self._x = np.zeros(NUMVARS)

        self._x[iX] = initial_x
        self._x[iV] = initial_v

        self._accel_variance = accel_variance

        # covariance of state GRV
        self._P = np.eye(NUMVARS)

    def predict(self, dt: float) -> None:
        # x = F x
        # P = F P Ft + G Gt a
        F = np.eye(NUMVARS)
        F[iX, iV] = dt
        new_x = F.dot(self._x)

        G = np.zeros((2, 1))
        G[iX] = 0.5 * dt ** 2
        G[iV] = dt
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance

        self._P = new_P
        self._x = new_x

    def update(self, meas_value: float, meas_variance: float):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        H = np.zeros((1, NUMVARS))
        H[0, iX] = 1

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_x

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        return self._x[iX]

    @property
    def vel(self) -> float:
        return self._x[iV]
#monitoring and alerting

#Guidance

