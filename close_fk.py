import sympy as sy
from sympy import cos, sin
q1, q2, q3, l0x, l0y, l1x, l1y, l2, l3 = sy.symbols("q1 q2 q3 l0x l0y l1x l1y l2 l3") 

T01 = sy.Matrix([[1, 0, 0, l0x],
                  [0, cos(q1), -sin(q1), l0y],
                  [0, sin(q1), cos(q1), 0], 
                  [0, 0, 0,1]])

T12 = sy.Matrix([[cos(q2), 0, sin(q2), l1x],
                  [0, 1, 0, l1y],
                  [-sin(q2), 0, cos(q2), 0], 
                  [0, 0, 0,1]])

T23 = sy.Matrix([[cos(q3), 0, sin(q3), 0],
                  [0, 1, 0, 0],
                  [-sin(q3), 0, cos(q3), -l2], 
                  [0, 0, 0,1]])

T34 = sy.Matrix([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, -l3], 
                  [0, 0, 0,1]])

print(sy.trigsimp(T01*T12*T23*T34))
"""
Matrix([[cos(q2 + q3), 0, sin(q2 + q3), l0x + l1x - l2*sin(q2) - l3*sin(q2 + q3)],
 [sin(q1)*sin(q2 + q3), cos(q1), -sin(q1)*cos(q2 + q3), l0y + l1y*cos(q1) + l2*sin(q1)*cos(q2) + l3*sin(q1)*cos(q2 + q3)],
 [-sin(q2 + q3)*cos(q1), sin(q1), cos(q1)*cos(q2 + q3), l1y*sin(q1) - l2*cos(q1)*cos(q2) - l3*cos(q1)*cos(q2 + q3)],
 [0, 0, 0, 1]])
"""