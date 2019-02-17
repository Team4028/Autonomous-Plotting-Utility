import math as m
import numpy as np

class limeLightData:
    def __init__(self, length, angleOne, angleTwo):
        self.l = length
        self.a1 = angleOne
        self.a2 = angleTwo
    
    def getIsometry(self, heading, targetAngle):
        h = np.deg2rad(heading)
        ta = np.deg2rad(targetAngle)
        A1 = np.deg2rad(self.a1)
        dx = self.l * m.cos(A1 + h)
        dy = self.l * m.sin(A1 + h)
        dtheta = ta - A1 - h
        return isometry(translation(dx, dy), rotation(dtheta))        

class translation:
    def __init__(self, ex, why):
        self.x = ex
        self.y = why
    
    def translateBy(self, other):
        return translation(self.x + other.x, self.y + other.y)
    
    def rotateBy(self, rot):
        return translation(self.x * rot.cos() - self.y * rot.sin(), self.x * rot.sin() + self.y * rot.cos())
    
class rotation:
    def __init__(self, meas, radQ = True):
        if radQ:
            self.theta = meas
        else:
            self.theta = np.deg2rad(meas)
    
    def cos(self):
        return m.cos(self.theta)
    
    def sin(self):
        return m.sin(self.theta)
    
    def rotateBy(self, other):
        return rotation(self.theta + other.theta)
    
class isometry:
    def __init__(self, translation, rotation):
        self.rot = rotation
        self.trans = translation
    
    def transformBy(self, other):
        return isometry(self.trans.translateBy(other.trans.rotateBy(self.rot)), self.rot.rotateBy(other.rot))
    
    def getLimeLightData(self, a2, targetAngle, heading):
        dx = self.trans.x
        dy = self.trans.y
        l = m.sqrt(dx * dx + dy * dy)
        a1 = targetAngle - heading - np.rad2deg(self.rot.theta)
        return limeLightData(l, a1, a2)
        
    
def getModifiedLimeLightData(angle1Deg, angle2Deg, lengthInches, headingDegrees, targetAngleDegrees, llxOffsetInches, llyOffsetInches, llthetaOffsetDegrees):
    limeLightToTargetData = limeLightData(lengthInches, angle1Deg, angle2Deg)
    limeLightToTargetIsometry = limeLightToTargetData.getIsometry(headingDegrees, targetAngleDegrees)
    vehicleToLimeLightIsometry = isometry(translation(llxOffsetInches, llyOffsetInches), rotation(llthetaOffsetDegrees, False))
    vehicleToTargetIsometry = vehicleToLimeLightIsometry.transformBy(limeLightToTargetIsometry)
    vehicleToTargetData = vehicleToTargetIsometry.getLimeLightData(angle2Deg, targetAngleDegrees, headingDegrees)
    return vehicleToTargetData

################ Params ######################
ll_x_offset_inches = 9
ll_y_offset_inches = -8
ll_theta_offset_degrees = 0
distance_to_target_inches = 100
angle_one_degrees = 0
angle_two_degrees = 0
heading_degrees = 90
target_angle_degrees = 90
###############################################

def main():
    veh2TargData = getModifiedLimeLightData(angle_one_degrees, angle_two_degrees, distance_to_target_inches, heading_degrees, target_angle_degrees, ll_x_offset_inches, ll_y_offset_inches, ll_theta_offset_degrees)
    print("New Distance: " + str(veh2TargData.l))
    print("New Angle One: " + str(veh2TargData.a1))
    print("New Angle Two: " + str(veh2TargData.a2))

main()
    