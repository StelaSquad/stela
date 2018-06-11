import numpy as np
from datetime import datetime as dt
import ephem

class Transform():

    def __init__(self):
        """
        Class for transforming between celestial and altitude azimuth coordinate systems.
        """
        self.refdate = 43830.0
        self.refcoors = [280.36317273, -0.01997193]
        
    def earthnorm2icrs(self,loc,time=None):

        if time == None:
            time = ephem.now()

        # Calculate siderial days
        sidereal_solar = 1/0.99726966323716
        t = (ephem.date(time) - self.refdate) * sidereal_solar

        # Calculate how the earth rotated since reference coordinate
        earthrotation = t % 1 * 360 + self.refcoors[0]
        
        # Estimate adjustments due to polar motions and precession
        adjlon = 0.00575 * (np.cos(2*t*np.pi/sidereal_solar + 0.01) - 1)
        adjlat = (1 + 1.2 * t/10000) * 0.11 * np.cos(2*t*np.pi + 1.755)
        
        # Rotate longitude by the given amount and add in adjustmente
        lon = loc[0] + earthrotation + adjlon
        lon = lon%360
        lat = loc[1] + adjlat

        return np.array([lon,lat])

        
    def cel2altaz(self,celcoor,loc,obstime="now"):
        
        if obstime == "now":
            obstime = ephem.date(dt.utcnow())
        
        # Calculate earth normal vector in celestial coordinates
        norm = self.earthnorm2icrs(loc,obstime)
        
        # Transform to altaz frame
        azalt = self.vec_prime(norm[0],norm[1],self.cartesian(celcoor),form="az-alt")
        
        return np.array([azalt])
        #np.array([a*180/np.pi for a in azalt])
    
    
    def cartesian(self,coor,unit="deg"):
        
        if unit == "deg":
            c = [co*np.pi/180 for co in coor]
        else:
            c = coor
           
        # Calculate cartesian coordinates
        xyz = [np.cos(c[0])*np.cos(c[1]),
               np.sin(c[0])*np.cos(c[1]),
               np.sin(c[1])]

        return xyz
    
    
    def xp(self,a,b,unit="deg"):
        """
        Returns x-prime, the vector representing the primed x-axis in the celestial frame. 
        Points toward the sky, normal to surface of earth.
        
        Input:
            ------
            a: float 
                Alpha, the angle that the primed coordinate system is spun about its z-axis.
               
            b: float
                Beta, angle that the primed coordinate system is pitched about its y-axis.
                
        Output:
            ------
            x_prime: ndarray [arg, dtype=float, shape=(3)
                3-vector that represents the x-axis of the prime frame in celestial xyz coordinates.
        """
        if unit == "deg":
            a *= np.pi/180
            b *= np.pi/180
            
        x_prime = np.array([np.cos(a)*np.cos(b),np.sin(a)*np.cos(b),np.sin(b)])
        return x_prime

    def yp(self,a,b,unit="deg"):
        """
        Returns y-prime, the vector representing the primed x-axis in the celestial frame. 
        Points toward the east, tangent to surface of earth.
        
        Input:
            ------
            a: float 
                Alpha, the angle that the primed coordinate system is spun about its z-axis.
               
            b: float
                Beta, angle that the primed coordinate system is pitched about its y-axis.
                
        Output:
            ------
            y_prime: ndarray [arg, dtype=float, shape=(3)
                3-vector that represents the y-axis of the prime frame in celestial xyz coordinates.
        """
        
        if unit == "deg":
            a *= np.pi/180
            b *= np.pi/180
        
        y_prime = np.array([-np.sin(a),np.cos(a),0])
        return y_prime
    
    def zp(self,a,b,unit="deg"):
        """Returns z-prime, the vector representation of the primed z-axis in the celestial frame.
        Points north, tangent to surface of earth
        
        Input:
            ------
            a: float 
                Alpha, the angle that the primed coordinate system is spun about its z-axis.
               
            b: float
                Beta, angle that the primed coordinate system is pitched about its y-axis.
                
        Output:
            ------
            z_prime: ndarray [arg, dtype=float, shape=(3)
                3-vector that represents the z-axis of the prime frame in celestial xyz coordinates.
        """
        if unit == "deg":
            a *= np.pi/180
            b *= np.pi/180
        
        z_prime = np.array([-np.cos(a)*np.sin(b),-np.sin(a)*np.sin(b),np.cos(b)])
        return z_prime


    def vec_prime(self, a, b, v, form='az-alt'):
        """
        Transforms a vector in equatorial celestial frame into an earth alt-az frame at location [a,b].
        
        Inputs:
            ------
            a: float 
                Alpha, the angle that the primed coordinate system is spun about its z-axis.
               
            b: float
                Beta, angle that the primed coordinate system is pitched about its y-axis.
                
            v: ndarray or list [arg, dtype = float, shape=(2) or shape=(3)]
                A celestial vector, in either a xyz or theta-phi representation.
                
        Optional:
            ------
            rep: string, arg = 'xyz' or 'az-alt'
                Specify whether the output data should be in xyz coordinates or azimuth altitude
                coordinates. 
                
        Output:
            v_prime: ndarray [arg, dtype=float, shape=(3) or shape=(2)]
                The vector represented in the primed coordinate system.
        """
        v = np.array(v)
        
        if len(v) == 2:
            v = self.cartesian(v)
            
            
        x = np.dot(self.xp(a,b),v)
        z = np.dot(self.zp(a,b),v)
        y = np.dot(self.yp(a,b),v)
        
        v_xyz = np.round(np.array([x,y,z]),12)

        
        if form == 'xyz':
            v_out = v_xyz
            
        elif form == 'az-alt':
            az = np.arctan2(v_xyz[1],v_xyz[2]) * 180/np.pi
            alt = np.arcsin(v_xyz[0]) * 180/np.pi
            
            if az < 0:
                az += 360
            
            v_out = np.array([az,alt])
            
        else:
            raise ValueError("Requested representation not understood. Use either 'xyz' or 'az-alt")
        
        return v_out
    
    
    def separation(self,coor1,coor2):
        
        # Convert both coordinates to cartesian
        xyz1 = self.cartesian(coor1)
        xyz2 = self.cartesian(coor2)
        
        # Calculate the dot product between them
        dot_prod = sum([xyz1[i]*xyz2[i] for i in [0,1,2]])
        
        # Sonce both magnitudes are 1, the dot product is only the cosine of the angle between both vectors
        ang = np.arccos(dot_prod) * 180/np.pi
        
        return round(ang,12)
        
        