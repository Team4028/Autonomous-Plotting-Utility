import numpy as np
import math as m
from random import random as rand
from timeit import default_timer as tic


#### Params ######
rhoTurning = 15


class point:
    def __init__(self, ex, why):
        self.x = ex
        self.y = why
        
    def _print(self):
        print("[" + str(self.x) + ", " + str(self.y) + "]")
class path:
    def __init__(self, segs):
        self.segments = segs
    
    def get_arc_len(self):
        tot_len = 0
        for seg in self.segments:
            tot_len += seg.arc_len()
        return tot_len

class geometry:
    def dist(p1, p2):
        return m.sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y))
    
    def get_intersect(x0, y0, t0, x1, y1, t1):
        x = (y0 - y1 + x1 * m.tan(t1) - x0 * m.tan(t0))/(m.tan(t1) - m.tan(t0))
        y = y0 + m.tan(t0)*(x - x0)
        return point(x, y)
    
class interval:
    def __init__(self, lower, upper):
        self.a = min(lower, upper)
        self.b = max(lower, upper)
        
    def get_length(self):
        return self.b - self.a
    
    def get_along(self, t):
        return self.a + t * self.get_length()
    
    def get_midpoint(self):
        return self.get_along(1/2)
    
    def bisect(self):
        return interval(self.a, self.get_midpoint()), interval(self.get_midpoint, self.b)
    
    def from_c_k_h(c, h, k):
        return interval( (c)/(2 ** k), (c + h)/(2 ** k))
    
    def is_inside(self, val):
        return not (val > self.b or val < self.a)

        
class line_segment:
    def __init__(self, p1, p2):
        self.starting_point = p1
        self.ending_point = p2
    
    def get_from_t(self, t):
        x_val = self.starting_point.x + t * (self.ending_point.x - self.starting_point.x)
        y_val = self.starting_point.y + t * (self.ending_point.y - self.starting_point.y)
        return point(x_val, y_val)
    
    def arc_len(self):
        x1 = self.starting_point.x
        x2 = self.ending_point.x
        dx = x2 - x1
        y1 = self.starting_point.y
        y2 = self.ending_point.y
        dy = y2 - y1
        return m.sqrt(dx * dx + dy * dy)
        
class quadratic_bezier:
    def __init__(self, p1, p2, p3):
        self.starting_point = p1
        self.middle_point = p2
        self.ending_point = p3
        
    def get_from_t(self, t):
        point_along_first_seg = line_segment(self.starting_point, self.middle_point).get_from_t(t)
        point_along_second_seg = line_segment(self.middle_point, self.ending_point).get_from_t(t)
        point_val = line_segment(point_along_first_seg, point_along_second_seg).get_from_t(t)
        return point_val
    
    def approx_with_segs(self, numSegs):
        segments = []
        t_vals = np.linspace(0.0, 1.0, num = numSegs)
        for ind in range(0, numSegs - 1):
            segments.append(line_segment(self.get_from_t(t_vals[ind]), self.get_from_t(t_vals[ind + 1])))
        return path(segments)
    
class cubic_bezier:
    def __init__(self, p1, p2, p3, p4):
        self.first_point = p1
        self.second_point = p2
        self.third_point = p3
        self.fourth_point = p4
    
    def get_from_t(self, t):
        point_along_first_seg = line_segment(self.first_point, self.second_point).get_from_t(t)
        point_along_second_seg = line_segment(self.second_point, self.third_point).get_from_t(t)
        point_along_third_seg = line_segment(self.third_point, self.fourth_point).get_from_t(t)
        point_val = quadratic_bezier(point_along_first_seg, point_along_second_seg, point_along_third_seg).get_from_t(t)
        return point_val
    
    def approx_with_segs(self, numSegs = 100):
        segments = []
        t_vals = np.linspace(0.0, 1.0, num = numSegs)
        for ind in range(0, numSegs - 1):
            segments.append(line_segment(self.get_from_t(t_vals[ind]), self.get_from_t(t_vals[ind + 1])))
        return path(segments)
    
    def get_analytic_arc_len(self):
        x0 = self.first_point.x
        x1 = self.second_point.x
        x2 = self.third_point.x
        x3 = self.fourth_point.x
        y0 = self.first_point.y
        y1 = self.second_point.y
        y2 = self.third_point.y
        y3 = self.fourth_point.y
        Ax = -3 * x0 + 9 * x1 - 9 * x2 + 3 * x3
        Bx = 6 * x0 - 12 * x1 + 6 * x2
        Cx = -3 * x0 + 3 * x1
        Ay = -3 * y0 + 9 * y1 - 9 * y2 + 3 * y3
        By = 6 * y0 - 12 * y1 + 6 * y2
        Cy = -3 * y0 + 3 * y1
        alpha = Ax * Ax + Ay * Ay
        beta = 2 * Ax * Bx + 2 * Ay * By
        gamma = 2 * Ax * Cx + Bx * Bx + 2 * Ay * Cy + By * By
        delta = 2 * Bx * Cx + 2 * By * Cy
        epsilon = Cx * Cx + Cy * Cy
        def _get_d_arc_len_d_t(t):
            return m.sqrt(alpha * t * t * t * t + beta * t * t * t + gamma * t * t + delta * t + epsilon)
        tot_len = calculus.nRiemannTrapezoidIntegrate(_get_d_arc_len_d_t, 0, 1)
        return tot_len
    
    def get_analytic_kurvature(self, t):
        x0 = self.first_point.x
        x1 = self.second_point.x
        x2 = self.third_point.x
        x3 = self.fourth_point.x
        y0 = self.first_point.y
        y1 = self.second_point.y
        y2 = self.third_point.y
        y3 = self.fourth_point.y
        Ax = -3 * x0 + 9 * x1 - 9 * x2 + 3 * x3
        Bx = 6 * x0 - 12 * x1 + 6 * x2
        Cx = -3 * x0 + 3 * x1
        Ay = -3 * y0 + 9 * y1 - 9 * y2 + 3 * y3
        By = 6 * y0 - 12 * y1 + 6 * y2
        Cy = -3 * y0 + 3 * y1
        alpha = Ax * Ax + Ay * Ay
        beta = 2 * Ax * Bx + 2 * Ay * By
        gamma = 2 * Ax * Cx + Bx * Bx + 2 * Ay * Cy + By * By
        delta = 2 * Bx * Cx + 2 * By * Cy
        epsilon = Cx * Cx + Cy * Cy
        kurvature = ((Ax * t * t + Bx * t + Cx) * (2 * Ay * t + By) - (2 * Ax *t + Bx) * (Ay * t * t + By * t + Cy))/((alpha * (t ** 4) + beta * (t ** 3) + gamma * (t ** 2) + delta * t + epsilon) ** 1.5)
        return kurvature
    
    def get_analytic_init_kurvature(self):
        x0 = self.first_point.x
        x1 = self.second_point.x
        x2 = self.third_point.x
        y0 = self.first_point.y
        y1 = self.second_point.y
        y2 = self.third_point.y
        Bx = 6 * x0 - 12 * x1 + 6 * x2
        Cx = -3 * x0 + 3 * x1
        By = 6 * y0 - 12 * y1 + 6 * y2
        Cy = -3 * y0 + 3 * y1
        epsilon = Cx * Cx + Cy * Cy
        kurvature = (Cx * By - Bx * Cy)/( epsilon ** 1.5)
        return kurvature
    
    def get_analytic_final_kurvature(self):
        x0 = self.first_point.x
        x1 = self.second_point.x
        x2 = self.third_point.x
        x3 = self.fourth_point.x
        y0 = self.first_point.y
        y1 = self.second_point.y
        y2 = self.third_point.y
        y3 = self.fourth_point.y
        Ax = -3 * x0 + 9 * x1 - 9 * x2 + 3 * x3
        Bx = 6 * x0 - 12 * x1 + 6 * x2
        Cx = -3 * x0 + 3 * x1
        Ay = -3 * y0 + 9 * y1 - 9 * y2 + 3 * y3
        By = 6 * y0 - 12 * y1 + 6 * y2
        Cy = -3 * y0 + 3 * y1
        alpha = Ax * Ax + Ay * Ay
        beta = 2 * Ax * Bx + 2 * Ay * By
        gamma = 2 * Ax * Cx + Bx * Bx + 2 * Ay * Cy + By * By
        delta = 2 * Bx * Cx + 2 * By * Cy
        epsilon = Cx * Cx + Cy * Cy
        kurvature = ((Ax + Bx + Cx) * (2 * Ay + By) - (2 * Ax + Bx) * (Ay + By + Cy))/((alpha + beta + gamma + delta + epsilon) ** 1.5)
        return kurvature
    
    
    
    def get_analytic_kurvature_prime(self, t):
        x0 = self.first_point.x
        x1 = self.second_point.x
        x2 = self.third_point.x
        x3 = self.fourth_point.x
        y0 = self.first_point.y
        y1 = self.second_point.y
        y2 = self.third_point.y
        y3 = self.fourth_point.y
        Ax = -3 * x0 + 9 * x1 - 9 * x2 + 3 * x3
        Bx = 6 * x0 - 12 * x1 + 6 * x2
        Cx = -3 * x0 + 3 * x1
        Ay = -3 * y0 + 9 * y1 - 9 * y2 + 3 * y3
        By = 6 * y0 - 12 * y1 + 6 * y2
        Cy = -3 * y0 + 3 * y1
        alpha = Ax * Ax + Ay * Ay
        beta = 2 * Ax * Bx + 2 * Ay * By
        gamma = 2 * Ax * Cx + Bx * Bx + 2 * Ay * Cy + By * By
        delta = 2 * Bx * Cx + 2 * By * Cy
        epsilon = Cx * Cx + Cy * Cy
        constant_one = 2 * Bx * Ay * t - 2 * Ax * By * t + 2 * Cx * Ay  - 2 * Ax * Cy
        constant_two = alpha * t * t * t * t + beta * t * t * t + gamma * t * t + delta * t + epsilon
        constant_three = 4 * alpha * (t ** 3) + 3 * beta * (t ** 2) + 2 * gamma * t + delta
        constant_four = Bx * Ay * t * t + 2 * Cx * Ay * t + Cx * By - Ay * Bx * t * t - 2 * Ax * Cy * t - Bx * Cy
        composite_constant_one = constant_one * (constant_two ** 1.5)
        composite_constant_two = 1.5 * m.sqrt(m.fabs(constant_two)) * constant_three * constant_four
        composite_constant_three = constant_two ** 3
        d_kurvature_d_t = (composite_constant_one - composite_constant_two)/composite_constant_three
        return d_kurvature_d_t   

    def get_max_squared_kurvature(self):
        x0 = self.first_point.x
        x1 = self.second_point.x
        x2 = self.third_point.x
        x3 = self.fourth_point.x
        y0 = self.first_point.y
        y1 = self.second_point.y
        y2 = self.third_point.y
        y3 = self.fourth_point.y
        Ax = -3 * x0 + 9 * x1 - 9 * x2 + 3 * x3
        Bx = 6 * x0 - 12 * x1 + 6 * x2
        Cx = -3 * x0 + 3 * x1
        Ay = -3 * y0 + 9 * y1 - 9 * y2 + 3 * y3
        By = 6 * y0 - 12 * y1 + 6 * y2
        Cy = -3 * y0 + 3 * y1
        alpha = Ax * Ax + Ay * Ay
        beta = 2 * Ax * Bx + 2 * Ay * By
        gamma = 2 * Ax * Cx + Bx * Bx + 2 * Ay * Cy + By * By
        delta = 2 * Bx * Cx + 2 * By * Cy
        epsilon = Cx * Cx + Cy * Cy
        velo_poly = polynomial([epsilon, delta, gamma, beta, alpha])
        null_points = numerical_methods.solvePolyInUnitInt(velo_poly)
        if len(null_points) != 0:
            return null_points[0], m.inf
        else:
            poly_one = polynomial([2 * Cx * Ay - 2 * Ax * Cy, 2 * Bx * Ay - 2 * Ax * By])
            composite_poly_one = polynomial.mult(poly_one, velo_poly)
            poly_two = velo_poly.derivative()
            poly_three = polynomial([Cx * By - Bx * Cy, 2 * Cx * Ay- 2 * Ax * Cy, Bx * Ay - Ax * By])
            composite_poly_two = polynomial.mult(poly_two, poly_three).scalar_multiply(-1.5)
            derivKurvPoly = polynomial.add(composite_poly_one, composite_poly_two)
            poss_vals = numerical_methods.solvePolyInUnitInt(derivKurvPoly)
            poss_vals.append(0)
            poss_vals.append(1)
            val_results = []
            for ind in range(0, len(poss_vals)):
                val_results.append(self.get_analytic_kurvature(poss_vals[ind]) ** 2)
            max_val = max(val_results)
            return poss_vals[val_results.index(max_val)], max_val
        
    def _print_control_points(self):
        self.first_point._print()
        self.second_point._print()
        self.third_point._print()
        self.fourth_point._print()
                         
        
    
class calculus:
    def nRiemannTrapezoidIntegrate(function, lower, upper, numTraps = 10000):
        area = 0
        dx = (upper - lower)/numTraps
        for i in range(0, numTraps - 1):
            leftVal = lower + i * dx
            rightVal = leftVal + dx
            leftHeight = function(leftVal)
            rightHeight = function(rightVal)
            averageHeight = (leftHeight + rightHeight)/2
            dArea = averageHeight * dx
            area += dArea
        return area
    
    def nDerivative(function, val, epsilon = .000001):
        return (function(val + epsilon) - function(val))/(epsilon)
    
    def nSecondDerivative(function, val, epsilon = .000001):
        def _local_derivative(_loc_val):
            return calculus.nDerivative(function, _loc_val, epsilon)
        return calculus.nDerivative(_local_derivative, val, epsilon)
    
    def nCurvature(xFunction, yFunction, t_val, epsilon = .000001):
        xPrime = calculus.nDerivative(xFunction, t_val, epsilon)
        yPrime = calculus.nDerivative(yFunction, t_val, epsilon)
        xPrimePrime = calculus.nSecondDerivative(xFunction, t_val, epsilon)
        yPrimePrime = calculus.nSecondDerivative(yFunction, t_val, epsilon)
        kurvature = (xPrime * yPrimePrime - xPrimePrime * yPrime)/((xPrime * xPrime + yPrime * yPrime) ** 1.5)
        return kurvature
    
    def nDerivCurvature(xFunction, yFunction, t_val, epsilon = .000001):
        def _local_curvature(t):
            return calculus.nCurvature(xFunction, yFunction, t, epsilon)
        return calculus.nDerivative(_local_curvature, t_val, epsilon)
    
class Three_d_Matrix:
    def __init__(self, array):
        self.arr = array
    
    def noneArr():
        return [[None, None, None], [None, None, None], [None, None, None]]
    
    def Two_d_det(a, b, c, d):
        return a * d - b * c
    
    def scalar_multiply(self, _lambda):
        newArr = Three_d_Matrix.noneArr()
        for i in range(0, 3):
            for j in range(0, 3):
                newArr[i][j] = _lambda * self.arr[i][j]
        return Three_d_Matrix(newArr)
    
    def transpose(self):
        newArr = Three_d_Matrix.noneArr()
        for i in range(0, 3):
            for j in range(0, 3):
                newArr[i][j] = self.arr[j][i]
        return Three_d_Matrix(newArr)
    
    def mat_of_minors(self):
        newArr = Three_d_Matrix.noneArr()
        newArr[0][0] = Three_d_Matrix.Two_d_det(self.arr[1][1], self.arr[1][2], self.arr[2][1], self.arr[2][2])
        newArr[0][1] = Three_d_Matrix.Two_d_det(self.arr[1][0], self.arr[1][2], self.arr[2][0], self.arr[2][2])
        newArr[0][2] = Three_d_Matrix.Two_d_det(self.arr[1][0], self.arr[1][1], self.arr[2][0], self.arr[2][1])
        newArr[1][0] = Three_d_Matrix.Two_d_det(self.arr[0][1], self.arr[0][2], self.arr[2][1], self.arr[2][2])
        newArr[1][1] = Three_d_Matrix.Two_d_det(self.arr[0][0], self.arr[0][2], self.arr[2][0], self.arr[2][2])
        newArr[1][2] = Three_d_Matrix.Two_d_det(self.arr[0][0], self.arr[0][1], self.arr[2][0], self.arr[2][1])       
        newArr[2][0] = Three_d_Matrix.Two_d_det(self.arr[0][1], self.arr[0][2], self.arr[1][1], self.arr[1][2])
        newArr[2][1] = Three_d_Matrix.Two_d_det(self.arr[0][0], self.arr[0][2], self.arr[1][0], self.arr[1][2])
        newArr[2][2] = Three_d_Matrix.Two_d_det(self.arr[0][0], self.arr[0][1], self.arr[1][0], self.arr[1][1])   
        return Three_d_Matrix(newArr)
    
    def cofactor_matrix(self):
        newArr = self.mat_of_minors().arr
        newArr[0][1] *= -1
        newArr[1][0] *= -1
        newArr[1][2] *= -1
        newArr[2][1] *= -1
        return Three_d_Matrix(newArr)
    
    def adjoint_matrix(self):
        return self.cofactor_matrix().transpose()
    
    def det(self):
        first_addend = self.arr[0][0] * Three_d_Matrix.Two_d_det(self.arr[1][1], self.arr[1][2], self.arr[2][1], self.arr[2][2])
        second_addend = self.arr[0][1] * Three_d_Matrix.Two_d_det(self.arr[1][0], self.arr[1][2], self.arr[2][0], self.arr[2][2])
        third_addend = self.arr[0][2] * Three_d_Matrix.Two_d_det(self.arr[1][0], self.arr[1][1], self.arr[2][0], self.arr[2][1])
        return first_addend - second_addend + third_addend
    
    def inverse(self):
        return self.adjoint_matrix().scalar_multiply(1/self.det())
    
    def vector_mult(self, vec):
        top = self.arr[0][0] * vec.arr[0] + self.arr[0][1] * vec.arr[1] + self.arr[0][2] * vec.arr[2]
        middle = self.arr[1][0] * vec.arr[0] + self.arr[1][1] * vec.arr[1] + self.arr[1][2] * vec.arr[2]
        bottom = self.arr[2][0] * vec.arr[0] + self.arr[2][1] * vec.arr[1] + self.arr[2][2] * vec.arr[2]
        return Three_d_Vector([top, middle, bottom])
    
class Three_d_Vector:
    def __init__(self, array):
        self.arr = array
    
    def magnitude(self):
        return m.sqrt(sum([k ** 2 for k in self.arr]))
    
    def scalar_mult(self, _lambda):
        return Three_d_Vector([k * _lambda for k in self.arr])
    
    def add(self, other):
        return Three_d_Vector([self.arr[i] + other.arr[i] for i in range(0, 3)])
    
        

class numerical_methods:
    """
    def isolate_roots_in_unit_int(poly):      #Uses The Bisection Algorithm to isolate roots in the unit interval assuming poly is square free
        L = [[0, 0, poly]]
        Isol = []
        n = poly.deg
        while len(L) != 0:
            for elem in L:
                L.remove(elem)
                if elem[2].evaluate(0) == 0:
                    elem[2] = elem[2].divX()
                    n -= 1
                    Isol.append([elem[0], elem[1], 0])
                v = polynomial.var(polynomial.compose(elem[2].reverse(), polynomial([1,1])).coeficientsList)
                if v == 1:
                    Isol.append([elem[0], elem[1], 1])
                elif v > 1:
                    L.append([2 * elem[0], elem[1] + 1, polynomial.compose(elem[2], polynomial([0, .5]).scalar_multiply(2 ** n))])
                    L.append([2 * elem[0] + 1, elem[1] + 1, polynomial.compose(elem[2], polynomial([.5, .5])).scalar_multiply(2 ** n)])
        return Isol
    
    def find_one_zero(function, interv, convDiff = .00001, iterMax = 500):            # Uses Brents algorithm to find a root in an interval
        leftVal = function(interv.a)
        rightVal = function(interv.b)
        if leftVal == 0:
            return interv.a
        elif rightVal == 0:
            return interv.b
        else:
            assert leftVal * rightVal < 0
        a = interv.a
        b = interv.b
        c = interv.a
        s = interv.b
        d = m.inf
        flagSet = True
        iterations = 0
        if m.fabs(leftVal) < m.fabs(rightVal):
            dummy = a
            a = b
            b = dummy
        while function(b) != 0 and function(s) != 0 and m.fabs(b-a )> convDiff and iterations < iterMax:
            iterations += 1
            if function(a) != function(c) and function(b) != function(c):
                s = (a * function(b) * function(c))/((function(a) - function(b)) * (function(a) - function(c))) + (b * function(a) * function(c))/((function(b) - function(a)) * (function(b) - function(c))) + (c * function(a) * function(b))/((function(c) - function(a)) * (function(c) - function(a)))
            else:
                s = b - function(b) * (b-a)/(function(b) - function(a))
            cond1 = not interval((3 * a + b)/4, b).is_inside(s) 
            cond2 = flagSet and (2 * m.fabs(s - b) >= m.fabs(b-c))
            cond3 = (not flagSet) and (2 * m.fabs(s - b) >= m.fabs(c - d))
            cond4 = flagSet and (m.fabs(b -c) < convDiff)
            cond5 = (not flagSet) and (m.fabs(c - d) < convDiff)
            if cond1 or cond2 or cond3 or cond4 or cond5:
                s = (a + b)/2
                flagSet = True
            else:
                flagSet = False
            d = c
            c = b
            if function(a)*function(s) < 0:
                b = s
            else:
                a = s
            if m.fabs(function(a)) < m.fabs(function(b)):
                dummy = a
                a = b
                b = dummy
        if m.fabs(function(s)) < m.fabs(function(b)):
            return s
        else:
            return b
    
    """
    def solvePolyinInt(poly, interv, kEpsilon = .000001):
        zeroes = np.roots(list(reversed(poly.coeficientsList)))
        working_sols = []
        for zero in zeroes:
            if m.fabs(zero.imag) < kEpsilon:
                if interv.is_inside(zero.real):
                    working_sols.append(zero.real)
        return working_sols
    
    def solvePolyInUnitInt(poly):
        return numerical_methods.solvePolyinInt(poly, interval(0, 1))
    """
    
    def solvePolyInUnitInt(poly):
        def _local_poly_eval_func(x):
                return poly.evaluate(x)
        sols = []
        possZeroes = numerical_methods.isolate_roots_in_unit_int(poly)
        for region in possZeroes:
            sols.append(numerical_methods.find_one_zero(_local_poly_eval_func, interval.from_c_k_h(region[0], region[1], region[2])))
        return sols  
    """
    
    def three_by_three_damped_Newtons_Method(initVector, vecValuedFunction, JacobianFunction, numSteps, c = 1, lambdaFac = .5):
        curX = initVector
        for i in range(0, numSteps):
            J = JacobianFunction(curX)
            f = vecValuedFunction(curX)
            dVec = J.inverse().vector_mult(f).scalar_mult(-1)
            while dVec.magnitude() > c * curX.magnitude():
                dVec = dVec.scalar_mult(lambdaFac)
            curX = curX.add(dVec)
        return curX
             
        
        
        
class polynomial:
    def __init__(self, coeficients):
        if coeficients is None:
            self.coeficientsList = [0]
            self.deg = 0
        else:
            self.coeficientsList = coeficients
            self.deg = len(self.coeficientsList) - 1
    def evaluate(self, x):
        val = 0
        for i in range(0, self.deg + 1):
            val += self.coeficientsList[i] * (x ** i)
        return val
    
    def reverse(self):
        return polynomial(list(reversed(self.coeficientsList)))
    
    def derivative(self):
        derivCoefs = [0] * self.deg
        for i in range(1, self.deg + 1):
            derivCoefs[i-1] = i * self.coeficientsList[i]
        return polynomial(derivCoefs)
    
    def evaluate_derivative(self, x):
        return self.derivative().evaluate()
    
    def scalar_multiply(self, _lambda):
        res = [0] * (self.deg + 1)
        for i in range(0, self.deg + 1):
            res[i] = self.coeficientsList[i] * _lambda
        return polynomial(res)
    
    def add(p1, p2):
        if p1.deg == p2.deg:
            res = [0] * (p1.deg + 1)
            for i in range(0, p1.deg + 1):
                res[i] = p1.coeficientsList[i] + p2.coeficientsList[i]
        elif p1.deg > p2. deg:
            res = [0] * (p1.deg + 1)
            for i in range(0, p2.deg +1):
                res[i] = p1.coeficientsList[i] + p2.coeficientsList[i]
            for j in range(p2.deg + 1, p1.deg + 1):
                res[j] = p1.coeficientsList[i]
        else:
            res = [0] * (p2.deg + 1)
            for i in range(0, p1.deg + 1):
                res[i] = p1.coeficientsList[i] + p2.coeficientsList[i]
            for j in range(p1.deg + 1, p2.deg + 1):
                res[j] = p2.coeficientsList[j]
        return polynomial(res)
    
    def mult(p1, p2):
        res = [0] * (p1.deg + p2.deg + 1)
        for i in range(0, p1.deg + 1):
            for j in range(0, p2.deg + 1):
                res[i + j] += p1.coeficientsList[i] * p2.coeficientsList[j]
        return polynomial(res)
    
    def Sigma(listOfPolys):
        if len(listOfPolys) == 0:
            return polynomial([0])
        elif len(listOfPolys) == 1:
            return listOfPolys[0]
        else:
            res = listOfPolys[0]
            for i in range(1, len(listOfPolys)):
                res = polynomial.add(res, listOfPolys[i])
            return res
        
    def Pi(listOfPolys):
        if len(listOfPolys) == 0:
            return polynomial([1])
        elif len(listOfPolys) == 1:
            return listOfPolys[0]
        else:
            res = listOfPolys[0]
            for i in range(1, len(listOfPolys)):
                res = polynomial.mult(res, listOfPolys[i])
            return res
            
    def power(self, p):
        if p == 0:
            return polynomial([1])
        else:
            return polynomial.Pi([self] * p)
    
    def compose(f, g):
        return polynomial.Sigma([g.power(i).scalar_multiply(f.coeficientsList[i]) for i in range(0, f.deg + 1)])
    
    def var(coefList):
        tot = 0
        for i in range(0, len(coefList) - 1):
            if m.copysign(1, coefList[i]) != m.copysign(1, coefList[i+1]):
                tot += 1
        return tot
    
    def divX(self):
        newList = self.coeficientsList
        newList.pop(0)
        return polynomial(newList)


class problem:
    def __init__(self, X0, Y0, THETA0, X1, Y1, THETA1):
        self.Xi = X0
        self.Yi = Y0
        self.THETAi = THETA0
        self.Xf = X1
        self.Yf = Y1
        self.THETAf = THETA1
    
    def gen_curve(self, r, s):
        return cubic_bezier(point(self.Xi, self.Yi), point(self.Xi + r * m.cos(self.THETAi), self.Yi + r * m.sin(self.THETAi)), point(self.Xf - s * m.cos(self.THETAf), self.Yf - s * m.sin(self.THETAf)), point(self.Xf, self.Yf))
    
    def l(self, r, s):
        return self.gen_curve(r, s).approx_with_segs().get_arc_len()
    
    def psi(self, r, s):
        return self.gen_curve(r, s).get_max_squared_kurvature()[1] - 1/ (rhoTurning * rhoTurning)   
    
    def l_r(self, r, s):
        def _local_function(t):
            return self.l(t, s)
        return calculus.nDerivative(_local_function, r)
    
    def l_s(self, r, s):
        def _local_function(t):
            return self.l(r, t)
        return calculus.nDerivative(_local_function, s)
    
    def l_rr(self, r, s):
        def _local_function(t):
            return self.l_r(t, s)
        return calculus.nDerivative(_local_function, r)
    
    def l_rs(self, r, s):
        def _local_function(t):
            return self.l_r(r, t)
        return calculus.nDerivative(_local_function, s)
    
    def l_sr(self, r, s):
        def _local_function(t):
            return self.l_s(t, s)
        return calculus.nDerivative(_local_function, r)
    
    def l_ss(self, r, s):
        def _local_function(t):
            return self.l_s(r, t)
        return calculus.nDerivative(_local_function, s)
    
    def psi_r(self, r, s):
        def _local_function(t):
            return self.psi(t, s)
        return calculus.nDerivative(_local_function, r)
    
    def psi_s(self, r, s):
        def _local_function(t):
            return self.psi(r, t)
        return calculus.nDerivative(_local_function, s)
    
    def psi_rr(self, r, s):
        def _local_function(t):
            return self.psi_r(t, s)
        return calculus.nDerivative(_local_function, r)
    
    def psi_rs(self, r, s):
        def _local_function(t):
            return self.psi_r(r, t)
        return calculus.nDerivative(_local_function, s)
    
    def psi_sr(self, r, s):
        def _local_function(t):
            return self.psi_s(t, s)
        return calculus.nDerivative(_local_function, r)
    
    def psi_ss(self, r, s):
        def _local_function(t):
            return self.psi_s(r, t)
        return calculus.nDerivative(_local_function, s)
    
    def get_candidate(self, startVec, numSteps = 100):
        def vecFunction(vec):
            r = vec.arr[0]
            s = vec.arr[1]
            mu = vec.arr[2]
            res = Three_d_Vector([mu * self.psi_r(r, s) + self.l_r(r,s), mu * self.psi_s(r, s) + self.l_s(r,s), mu * self.psi(r,s)])
            return res
        def jacobianFunction(vec):
            r = vec.arr[0]
            s = vec.arr[1]
            mu = vec.arr[2]
            arr = [[mu * self.psi_rr(r,s) + self.l_rr(r,s), mu * self.psi_rs(r, s) + self.l_rs(r, s), self.psi_r(r, s)], [mu * self.psi_sr(r, s) + self.l_sr(r, s), mu * self.psi_ss(r, s) + self.l_ss(r, s), self.psi_s(r, s)], [mu * self.psi_r(r,s), mu * self.psi_s(r,s), self.psi(r, s)]]
            return Three_d_Matrix(arr)
        return numerical_methods.three_by_three_damped_Newtons_Method(startVec, vecFunction, jacobianFunction,  numSteps)
    
    def init_candidate(self):
        if m.fabs(m.tan(self.THETAi)) != m.fabs(m.tan(self.THETAf)):
            intPoint = geometry.get_intersect(self.Xi, self.Yi, self.THETAi, self.Xf, self.Yf, self.THETAf)
            r_init = geometry.dist(point(self.Xi, self.Yi), intPoint) * .5
            s_init = geometry.dist(intPoint, point(self.Xf, self.Yf)) * .5
            return Three_d_Vector([r_init, s_init, 0])
        else:
            r_init = geometry.dist(point(self.Xi, self.Yi), point(self.Xf, self.Yf)) * .5
            s_init = geometry.dist(point(self.Xi, self.Yi), point(self.Xf, self.Yf)) * .5
            return Three_d_Vector([r_init, s_init, 0])
    
    def rand_cand(self):
        intPoint = geometry.get_intersect(self.Xi, self.Yi, self.THETAi, self.Xf, self.Yf, self.THETAf)
        r_init = rand() * 2 * geometry.dist(point(self.Xi, self.Yi), intPoint)
        s_init = rand() * 2 * geometry.dist(intPoint, point(self.Xf, self.Yf))
        mu = rand(0, .5)
        return Three_d_Vector([r_init, s_init, mu])
        
    def solve(self):
        vec = self.get_candidate(self.init_candidate()).arr
        return self.gen_curve(vec[0], vec[1])     
        
            
            
class testing:
    def test_arc_length_formulae():
        real_len = m.sqrt(2)
        path_len = cubic_bezier(point(0, 0), point(1/3, 1/3), point(2/3, 2/3), point(1, 1)).approx_with_segs().get_arc_len()
        analytic_len = cubic_bezier(point(0, 0), point(1/3, 1/3), point(2/3, 2/3), point(1, 1)).get_analytic_arc_len()
        print("Exact Arc Length:                         " + str(real_len))
        print("Line Segment Path Generated Arc Length:   " + str(path_len))
        print("Explicit Integration Arc Length:          " + str(analytic_len))
        
        #########################################################################################
        ####### OUTPUT #########################################################################
        #######################################################################################
        ######  Exact Arc Length:                         1.4142135623730951    ##############
        #####   Line Segment Path Generated Arc Length:   1.4142135623731005   ##############
        ####    Explicit Integration Arc Length:          1.414072141017049   ##############
        ###################################################################################
        ##################################################################################
        
    def test_kurvature_and_arc_length_formulae():
        t_val = .8344
        real_kurvature = .019185335534
        bezCurve = cubic_bezier(point(0, 0), point(0, 50), point(50, 50), point(50, 100))
        def get_x(t):
            return bezCurve.get_from_t(t).x
        def get_y(t):
            return bezCurve.get_from_t(t).y
        numeric_kurvature = calculus.nCurvature(get_x, get_y, t_val)
        analytic_kurvature = bezCurve.get_analytic_curvature(t_val)
        real_len = 115.551438857
        path_len = bezCurve.approx_with_segs().get_arc_len()
        analytic_len = bezCurve.get_analytic_arc_len()
        print("Curvature Approximations at t = " + str(t_val))
        print('')
        print("Exact Curvature:                          " + str(real_kurvature))
        print("Numerical Derivate Curvature:             " + str(numeric_kurvature))
        print("Explicit Integration Curvature:           " + str(analytic_kurvature))
        print('')
        print("Arc Length Approximations")
        print ('')
        print("Exact Arc Length                          " + str(real_len))
        print("Line Segment Path Generated Arc Length:   " + str(path_len))
        print("Explicit Integration Arc Length:          " + str(analytic_len))
        
        ##################################################################################################
        ####### OUTPUT ##################################################################################
        ################################################################################################
        #       Curvature Approximations at t = 0.8344                                                 #
        #
        #       Exact Curvature:                          0.019185335534
        #       Numerical Derivate Curvature:             0.019184989436457307
        #       Explicit Integration Curvature:           0.019185333553446548
        #       
        #       Arc Length Approximations
        #        
        #       Exact Arc Length                          115.551438857
        #       Line Segment Path Generated Arc Length:   115.55142385055224
        #       Explicit Integration Arc Length:          115.53644085700363
        ##########################################################################
        #########################################################################
        
    def test_init_kurvature_formula():
        t_val = 0
        real_kurvature = -1/75
        bezCurve = cubic_bezier(point(0, 0), point(0, 50), point(50, 50), point(50, 100))
        def get_x(t):
            return bezCurve.get_from_t(t).x
        def get_y(t):
            return bezCurve.get_from_t(t).y
        numeric_kurvature = calculus.nCurvature(get_x, get_y, t_val)
        analytic_kurvature = bezCurve.get_analytic_init_kurvature()
        print("Curvature Approximations at t = " + str(t_val))
        print('')
        print("Exact Curvature:                          " + str(real_kurvature))
        print("Numerical Derivate Curvature:             " + str(numeric_kurvature))
        print("Explicit Analytic Curvature:              " + str(analytic_kurvature))
        
        ##################################################################################################
        ####### OUTPUT ##################################################################################
        ################################################################################################
        #  Curvature Approximations at t = 0
        #
        #  Exact Curvature:                          -0.013333333333333334
        #  Numerical Derivate Curvature:             -0.013333346666619998
        #  Explicit Analytic Curvature:              -0.013333333333333334
        ###########################################################################################
        
    def test_final_kurvature_formula():
        t_val = 1
        real_kurvature = 1/75
        bezCurve = cubic_bezier(point(0, 0), point(0, 50), point(50, 50), point(50, 100))
        def get_x(t):
            return bezCurve.get_from_t(t).x
        def get_y(t):
            return bezCurve.get_from_t(t).y
        numeric_kurvature = calculus.nCurvature(get_x, get_y, t_val)
        analytic_kurvature = bezCurve.get_analytic_final_kurvature()
        print("Curvature Approximations at t = " + str(t_val))
        print('')
        print("Exact Curvature:                          " + str(real_kurvature))
        print("Numerical Derivate Curvature:             " + str(numeric_kurvature))
        print("Explicit Analytic Curvature:              " + str(analytic_kurvature))
        
        ##################################################################################################
        ####### OUTPUT ##################################################################################
        ################################################################################################     
        #  Curvature Approximations at t = 1
        #
        #  Exact Curvature:                          0.013333333333333334
        #  Numerical Derivate Curvature:             0.013332899695268371
        #  Explicit Analytic Curvature:              0.013333333333333334
        ###########################################################################################    
        
    def test_kurvature_derivative_formula():
        t_val = .5
        real_deriv_kurvature = 4 * m.sqrt(2) / 75 
        bezCurve = cubic_bezier(point(0, 0), point(0, 50), point(50, 50), point(50, 100))
        def get_x(t):
            return bezCurve.get_from_t(t).x
        def get_y(t):
            return bezCurve.get_from_t(t).y
        numeric_deriv_kurvature = calculus.nDerivCurvature(get_x, get_y, t_val, epsilon = .0001) #For some reason, the 10^-6 epsilon was numerically instable
        analytic_deriv_kurvature = bezCurve.get_analytic_kurvature_prime(t_val)
        print("Derivative of Curvature Approximations at t = " + str(t_val))
        print('')
        print("Exact Derivative of Curvature:               " + str(real_deriv_kurvature))
        print("Numerical Derivate of Curvature:             " + str(numeric_deriv_kurvature))
        print("Explicit Analytic Defivative of Curvature:   " + str(analytic_deriv_kurvature))
        
        ##################################################################################################
        ####### OUTPUT ##################################################################################
        ################################################################################################           
        #  Derivative of Curvature Approximations at t = 0.5
        #
        #  Exact Derivative of Curvature:               0.07542472332656508
        #  Numerical Derivate of Curvature:             0.0754240596610785
        #  Explicit Analytic Defivative of Curvature:   0.07542472332656507
        ############################################################################################
        
    def test_ThreeD_Matrices():
        mat = Three_d_Matrix([[1, 0, 1],[0, 1, 0],[0, 0, 2]])
        vec = [1, .5, -2]
        actual_det = 2
        computed_det = mat.det()
        actual_inverse = [[1, 0, -.5],[0, 1, 0], [0, 0, .5]]
        computed_inverse = mat.inverse().arr
        actual_multiplication = [-1, .5, -4]
        computed_multiplication = mat.vector_mult(vec)
        print("Actual Determinant:               " + str(actual_det))
        print("Computed Determinant:             " + str(computed_det))
        print("Actual Inverse:                   " + str(actual_inverse))
        print("Computed Inverse:                 " + str(computed_inverse))
        print("Actual Vector Multiplication:     " + str(actual_multiplication))
        print("Computed Vector MultiplicationL   " + str(computed_multiplication))
        
        ##################################################################################################
        ####### OUTPUT ##################################################################################
        ################################################################################################       
        ##  Actual Determinant:               2
        ##  Computed Determinant:             2
        ##  Actual Inverse:                   [[1, 0, -0.5], [0, 1, 0], [0, 0, 0.5]]
        ##  Computed Inverse:                 [[1.0, 0.0, -0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]]
        ##  Actual Vector Multiplication:     [-1, 0.5, -4]
        ##  Computed Vector MultiplicationL   [-1.0, 0.5, -4.0]
        ###############################################################################################
        
    def test_polynomials():
        actual_pol = [2, -2, 1]
        computed_pol = polynomial.compose(polynomial([1, 0, 1]), polynomial([-1, 1]))
        actual_var = 2
        computed_var = polynomial.var(computed_pol.coeficientsList)
        print("Actual Coefficients:      " + str(actual_pol))
        print("Computed Coefficients:    " + str(computed_pol.coeficientsList))
        print("Actual Var:               " + str(actual_var))
        print("Computed Var:             " + str(computed_var))
        
        ##################################################################################################
        ####### OUTPUT ##################################################################################
        ################################################################################################       
        ##   Actual Coefficients:      [2, -2, 1]
        ##   Computed Coefficients:    [2, -2, 1]
        ##   Actual Var:               2
        ##   Computed Var:             2
        ###########################################################################################
        
    def test_rootFinding():
        real_roots = [m.sqrt(2/3)]
        pol = polynomial([2, 0, -3])
        roots = numerical_methods.solvePolyInUnitInt(pol)
        print("Actual Roots in Unit Interval:     " + str(real_roots))
        print("Computed Roots in Unit Interval:   " + str(roots))
        
        ##################################################################################################
        ####### OUTPUT ##################################################################################
        ################################################################################################    
        ###  Actual Roots in Unit Interval:     [0.816496580927726]
        ###  Computed Roots in Unit Interval:   [0.816496580927726]
        ###############################################################################################
    
    def test_get_max_kurvature():
        real_t = .8344
        real_kappa_squared = .0192 ** 2
        bezCurve = cubic_bezier(point(0, 0), point(0, 50), point(50, 50), point(50, 100))
        computed_t, computed_kappa_squared = bezCurve.get_max_squared_kurvature()
        print("Actual Arg Max Squared Curvature:     " + str(real_t))
        print("Computed Arg Max Squared Curvature:   " + str(computed_t))
        print("Actual Max Squared Curvature:         " + str(real_kappa_squared))
        print("Computed Max Squared Curvature:       " + str(computed_kappa_squared))
        
        ##################################################################################################
        ####### OUTPUT ##################################################################################
        ################################################################################################            
        ##   Actual Arg Max Squared Curvature:     0.8344
        ##   Computed Arg Max Squared Curvature:   0.8343701524882139
        ##   Actual Max Squared Curvature:         0.00036863999999999994
        ##   Computed Max Squared Curvature:       0.00036807703333329925
        #############################################################################################
        
    def test_3d_Newtons_Method():
        def function(vec):
            x = vec.arr[0]
            y = vec.arr[1]
            z = vec.arr[2]
            return Three_d_Vector([x ** 2 + y ** 2 - z ** 2 - 10, x + y + z + 3, x ** 2 + 3 * x + y ** 2 - y + z ** 3 - 41])
        def Jacobian(vec):
            x = vec.arr[0]
            y = vec.arr[1]
            z = vec.arr[2]
            arr = [[2 * x, 2 * y, -2 * z], [1, 1, 1], [2 * x + 3, 2 * y - 1, 3 * z * z]]
            return Three_d_Matrix(arr)
        computedZero = numerical_methods.three_by_three_damped_Newtons_Method(Three_d_Vector([-2.25, -3.75, 3]), function, Jacobian, 100).arr
        actualZero = [-2.26316, -3.67857, 2.94173]
        print("Actual Solution Vector:    " + str(actualZero))
        print("Computed Solution Vector:  " + str(computedZero))
        
        ##################################################################################################
        ####### OUTPUT ##################################################################################
        ################################################################################################      
        ##  Actual Solution Vector:    [-2.26316, -3.67857, 2.94173]
        ##  Computed Solution Vector:  [-2.263159210058389, -3.678572639877363, 2.941731849935752]
        ############################################################################################
        
    def test_long_3d_Newtons_Method():
        def function(vec):
            x = vec.arr[0]
            y = vec.arr[1]
            z = vec.arr[2]
            return Three_d_Vector([x ** 2 + y ** 2 - z ** 2 - 10, x + y + z + 3, x ** 2 + 3 * x + y ** 2 - y + z ** 3 - 41])
        def Jacobian(vec):
            x = vec.arr[0]
            y = vec.arr[1]
            z = vec.arr[2]
            arr = [[2 * x, 2 * y, -2 * z], [1, 1, 1], [2 * x + 3, 2 * y - 1, 3 * z * z]]
            return Three_d_Matrix(arr)
        computedZero = numerical_methods.three_by_three_damped_Newtons_Method(Three_d_Vector([.1, .1, .1]), function, Jacobian, 100).arr
        actualZero = [-2.26316, -3.67857, 2.94173]
        print("Actual Solution Vector:    " + str(actualZero))
        print("Computed Solution Vector:  " + str(computedZero))
        
        ##################################################################################################
        ####### OUTPUT ##################################################################################
        ################################################################################################ 
        ###  Actual Solution Vector:    [-2.26316, -3.67857, 2.94173]
        ###  Computed Solution Vector:  [-2.2631592100583893, -3.678572639877363, 2.941731849935752]
        ##############################################################################################
        
    def test_very_long_3d_Newtons_Method():
        def function(vec):
            x = vec.arr[0]
            y = vec.arr[1]
            z = vec.arr[2]
            return Three_d_Vector([x ** 2 + y ** 2 - z ** 2 - 10, x + y + z + 3, x ** 2 + 3 * x + y ** 2 - y + z ** 3 - 41])
        def Jacobian(vec):
            x = vec.arr[0]
            y = vec.arr[1]
            z = vec.arr[2]
            arr = [[2 * x, 2 * y, -2 * z], [1, 1, 1], [2 * x + 3, 2 * y - 1, 3 * z * z]]
            return Three_d_Matrix(arr)
        computedZero = numerical_methods.three_by_three_damped_Newtons_Method(Three_d_Vector([100, 100, 100]), function, Jacobian, 100).arr
        actualZero = [-2.26316, -3.67857, 2.94173]
        print("Actual Solution Vector:    " + str(actualZero))
        print("Computed Solution Vector:  " + str(computedZero))
        
        ##################################################################################################
        ####### OUTPUT ##################################################################################
        ################################################################################################  
        ##   Actual Solution Vector:    [-2.26316, -3.67857, 2.94173]
        ##   Computed Solution Vector:  [-2.263159210058391, -3.6785726398773617, 2.9417318499357523]
        #################################################################################################
        
    def test_solution():
        prob = problem(0, 0, m.pi/2, 50, 100, m.pi*(17/12))
        sT = tic()
        sol = prob.solve()
        eT = tic()
        r = geometry.dist(sol.first_point, sol.second_point)
        s = geometry.dist(sol.third_point, sol.fourth_point)
        sol._print_control_points()
        prob.gen_curve(s, -1 * r)._print_control_points()
        print(eT - sT)
        

testing.test_solution()       

        

