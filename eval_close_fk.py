import numpy as np

from numpy import sin, cos


def close_fk(robot_type, joint_pos):
    assert robot_type in ["go1", "aliengo"]
    q1, q2, q3 = joint_pos
    if robot_type == "go1": 
        ################# go1 
        l0x = 0.1881
        l0y = 0.04675
        l1x = 0
        l1y = 0.08
        l2 = 0.213
        l3 = 0.213
        # q1, q2, q3 = 0.0218,  0.9071, -1.3261  #0.0169,  0.6291, -1.4231
        ################# 
    if robot_type == "aliengo":
        ################# aliengo
        l0x = 0.2399
        l0y = 0.051
        l1x = 0
        l1y = 0.083
        l2 = 0.25
        l3 = 0.25
        # q1, q2, q3 = 0., 0.6, 1.2
        #################

    foot_pos = np.array([[cos(q2 + q3), 0, sin(q2 + q3), l0x + l1x - l2*sin(q2) - l3*sin(q2 + q3)],
                         [sin(q1)*sin(q2 + q3), cos(q1), -sin(q1)*cos(q2 + q3), l0y + l1y*cos(q1) + l2*sin(q1)*cos(q2) + l3*sin(q1)*cos(q2 + q3)],
                         [-sin(q2 + q3)*cos(q1), sin(q1), cos(q1)*cos(q2 + q3), l1y*sin(q1) - l2*cos(q1)*cos(q2) - l3*cos(q1)*cos(q2 + q3)],
                         [0, 0, 0, 1]])
    return foot_pos[:3, 3]

def close_fk_go1(joint_pos):
    q1, q2, q3 = joint_pos
    l0x = 0.1881
    l0y = 0.04675
    l1x = 0
    l1y = 0.08
    l2 = 0.213
    l3 = 0.213
    foot_pos = np.array([[cos(q2 + q3), 0, sin(q2 + q3), l0x + l1x - l2*sin(q2) - l3*sin(q2 + q3)],
                         [sin(q1)*sin(q2 + q3), cos(q1), -sin(q1)*cos(q2 + q3), l0y + l1y*cos(q1) + l2*sin(q1)*cos(q2) + l3*sin(q1)*cos(q2 + q3)],
                         [-sin(q2 + q3)*cos(q1), sin(q1), cos(q1)*cos(q2 + q3), l1y*sin(q1) - l2*cos(q1)*cos(q2) - l3*cos(q1)*cos(q2 + q3)],
                         [0, 0, 0, 1]])
    return foot_pos[:3, 3]

def close_fk_aliengo(joint_pos):
    q1, q2, q3 = joint_pos
    l0x = 0.2399
    l0y = 0.051
    l1x = 0
    l1y = 0.083
    l2 = 0.25
    l3 = 0.25
    foot_pos = np.array([[cos(q2 + q3), 0, sin(q2 + q3), l0x + l1x - l2*sin(q2) - l3*sin(q2 + q3)],
                         [sin(q1)*sin(q2 + q3), cos(q1), -sin(q1)*cos(q2 + q3), l0y + l1y*cos(q1) + l2*sin(q1)*cos(q2) + l3*sin(q1)*cos(q2 + q3)],
                         [-sin(q2 + q3)*cos(q1), sin(q1), cos(q1)*cos(q2 + q3), l1y*sin(q1) - l2*cos(q1)*cos(q2) - l3*cos(q1)*cos(q2 + q3)],
                         [0, 0, 0, 1]])
    return foot_pos[:3, 3]



if __name__ == "__main__":
    robot_type = "aliengo"
    joint_pos = (0.0711,  0.6242, -1.8108)
    print(close_fk(robot_type, joint_pos))
