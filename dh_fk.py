import sympy as sy
from sympy import cos, sin

a, d, A, B = sy.symbols("a d A B")


TA = sy.Matrix([[1, 0, 0, 0],
                [0, cos(A), -sin(A), 0],
                [0, sin(A), cos(A), 0], 
                [0, 0, 0,1]])

TB = sy.Matrix([ [cos(B), -sin(B), 0,0],
                 [sin(B), cos(B), 0,0], 
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])


Td = sy.Matrix([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, d], 
                 [0, 0, 0,1]])

Ta = sy.Matrix([[1, 0, 0, a],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0], 
                 [0, 0, 0,1]])

# print(sy.trigsimp(TB*Td*Ta*TA))
# Matrix([[cos(B), -sin(B)*cos(A), sin(A)*sin(B), a*cos(B)], 
#         [sin(B), cos(A)*cos(B), -sin(A)*cos(B), a*sin(B)], 
#         [0, sin(A), cos(A), d],
#         [0, 0, 0, 1]])

# print(sy.trigsimp(Ta*TA*TB*Td))
# Matrix([[cos(B), -sin(B), 0, a],
#         [sin(B)*cos(A), cos(A)*cos(B), -sin(A), -d*sin(A)],
#         [sin(A)*sin(B), sin(A)*cos(B), cos(A), d*cos(A)],
#         [0, 0, 0, 1]])

print(sy.trigsimp(TA*Ta*TB*Td - Ta*TA*Td*TB))