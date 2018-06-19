import numpy as np
from datetime import datetime as dt
import ephem
from transform import Transform

class Triangulate():
    """
    This class is used to triangulate the location of the telescope based on three celestial objects. 
    
    It operates based on a latitude and longitude equatorial coordinate system, with celestial coordinates
    given in that frame. The telescope can be placed a floor that is not perfectly normal to the surface 
    of the earth. The use a (technically) false latitude and longitude, one for which the normal vector is the
    same as the normal vector of the base of the telescope, can account for this error. This is the method
    used, but note that the outputted earth coordinates will likely differ slightly than the true location of
    the telescope.
    
    Objects:
        Triangulate.obserrs: 
                Expected errors in measured altitude and azimuth (default=1')
        Triangulate.comp_power: T
                he calculative potential of the computer being used (default=100)
    
    Functions:
        Triangulate.triangulate:
                Computes the normal vector of the telescope based on three objects and their observed angular
                differences.
        
    Helper Functions:
        Triangulate.xp, Triangulate.yp, Triangulate.zp: 
                The coordinate vectors of a frame on the surface of the earth, at location [a,b].
        Triangulate.vec_prime: 
                Transforms a vector in equatorial frame into an earth alt-az frame at location [a,b].
        Triangulate.find_valid: 
                Given two objects and their angular difference, calculates probability the distribution across
                latitude and longitude space.
        Triangulate.match:
                Given one coordinate grid and three 2d probability distributions, calculates the total 
                probability distribution combining the three points.
        
    Outputs:
        Triangulate.lon: 
                Calculated longitude
        Triangulate.lat: 
                Calculated latitude
        Triangulate.errs: 
                The errors of the latitude and longitude [errlon,errlat]
    
    
    """
    def __init__(self):
        self.comp_power = 100
        self.chain=[]
        self.dist=[]
        self.dth=0
        self.dph=0
        self.obserrs = [np.pi/180/30,np.pi/180/30]
        self.classmod = 1
        
        self.frame = Transform()
    
    
    def gen_mock(self,loc=[],errs=[0,0]):
        """
        Generates a mock f data set, to test the program.
                
        Optional:
            ------
            errs: list or ndarray [args, dtype=float shape=(2)] 
                list the error in the the generated data of the observed angle differences
        
        Output:
            v_prime: ndarray [arg, dtype=float, shape=(3) or shape=(2)]
                The vector represented in the primed coordinate system.
        """
        
        if loc == []:
            # Generate random truth values
            a = np.random.rand()*360
            b = (np.random.rand()-0.5)*180*0.8
            
        else:
            [a,b] = loc
        
        
        # Generate three random triangulation vectors
        v1 = [np.random.uniform(0,360),np.random.uniform(-90,90)]
        v2 = [np.random.uniform(0,360),np.random.uniform(-90,90)]
        v3 = [np.random.uniform(0,360),np.random.uniform(-90,90)]
        
        # Calculate the true difference in alt-azimuth, and add a little random error
        v2_v1 = self.frame.vec_prime(a,b,v2) - self.frame.vec_prime(a,b,v1) + np.random.uniform(-errs[0],errs[0])
        v3_v2 = self.frame.vec_prime(a,b,v3) - self.frame.vec_prime(a,b,v2) + np.random.uniform(-errs[1],errs[1])
                
        
        return [a,b],v1,v2,v3,v2_v1,v3_v2
    

    def find_valid(self, obj1coor, obj2coor, obs, lims=[[0,360],[-90,90]]):
        """
        Calculates the probability distribution of lattitude and longitude points within the given limits.
        
        Inputs:
            ------
            obj1coor: list or ndarray [args, dtype=float, shape=(2)]
                Celestial angle coordinates for the first object
                
            obj2coor: list or ndarray [args, dtype=float, shape=(2)]
                Celestial angle coordinates for the second object
                
            obs: list or ndarray [args, dtype=float, shape=(2)]
                the observed difference in [azimuth, altitude] between object 1 and 2
                
        Optional:
            ------
            lims: list or ndarray [args, dtype=float, shape=(2,2)]
                The limits in which to look for valid longitude and lattitudes. Default is the entire space.
            
        Output:
            ------
            grid: list [args,dtype=float,shape=(2,100,100)]
                The coordinates of each longitude and lattitude tested
            
            dist_norm: ndarray [args, dtype=float, shape=(100,100)]
                The normalized probability distribution over all points in the grid.
                
        """
        
        #print obs*np.pi/180
        
        n=self.comp_power
    
        obsaz = obs[0]
        obsalt = obs[1]
        
        # Initalize variable space
        A_axis = np.linspace(lims[0][0],lims[0][1],n)
        B_axis = np.linspace(lims[1][0],lims[1][1],n)
        grid = np.meshgrid(A_axis,B_axis)

        # For each potential A and B vector, calculate the theoretical change in theta and phi
        
        # Function for calclating the azimuth and altitude angles differences for a given lon and lat
        diff_func = lambda lon,lat: (np.array(self.frame.vec_prime(lon,lat,obj2coor,form="az-alt")) -
                                np.array(self.frame.vec_prime(lon,lat,obj1coor,form="az-alt")))
        
        [az, alt] = np.array( map(diff_func, grid[0].flatten(), grid[1].flatten()) ).T
        
        #back from column to 2d grid
        az = np.array(az).reshape(n,n)
        alt = np.array(alt).reshape(n,n)

        # Start real small with the binsize, extremely restrictive
        stdaz = np.std(az.flatten())
        stdalt = np.std(alt.flatten())
        
        widaz = np.max(az) - np.min(az)
        widalt = np.max(alt) - np.min(alt)
        
        #print widaz, np.std(az)
        
        mod = (self.obserrs[0] + self.obserrs[1])/2
        
        # Algorithm for finding least possible errors that won't return an error
        #while mod*stdaz < 3*self.obserrs[0] and mod*stdalt < 3*self.obserrs[1]:    
        #    mod*=1.05
        
        dA = A_axis[1] - A_axis[0]
        dB = B_axis[1] - B_axis[0]
        
        mod = 1. / self.comp_power #* self.classmod
        
        mindistaz = np.min(np.abs(az-obsaz))
        mindistalt = np.min(np.abs(alt-obsalt))
        
        l_az = mindistaz*50*10
        l_alt = mindistalt*50*10
                
        # Turn distances into gaussian probability densities
        log_probs = - 0.5*(((az - obsaz)/(l_az))**2 + ((alt - obsalt)/(l_alt))**2)
        
        maxprob = np.exp(np.max(log_probs))
        
        probs_norm = np.exp(log_probs) / maxprob
        
        return grid,probs_norm

    def simple_filter(self,grid,probs):
        
        lon_ax = grid[0][0]
        lat_ax = grid[1][:,0]
        
        dlon = lon_ax[1] - lon_ax[0]
        dlat = lat_ax[1] - lat_ax[0]
        
        # Sum the n by n array upon it's axis to produce two 1d probability densities
        lon_hist = np.sum(probs,axis=0)
        lat_hist = np.sum(probs,axis=1)
        
        # Normalize the two probability distributions
        lon_hist /= (np.sum(lon_hist) * dlon)
        lat_hist /= (np.sum(lat_hist) * dlat)
        
        # Take the average
        lon_avg = np.sum(lon_hist*lon_ax) * dlon
        lat_avg = np.sum(lat_hist*lat_ax) * dlat
        
        # Take the standard deviation
        lon_std = np.sqrt( np.sum(lon_ax**2*lon_hist*dlon) - lon_avg**2 )
        lat_std = np.sqrt( np.sum(lon_ax**2*lon_hist*dlon) - lon_avg**2 )
        
        
        av = np.array([lon_avg,lat_avg])
        errs = np.array([lon_std,lat_std])
        
        
        # Finds the accuracy of the analysis, chooses new limits based on these
        r = 5

        errs += np.array([dlon,dlat])/(r)
        
        lims = np.array([av - r*errs, av + r*errs]).T
        
        return [lon_avg,lat_avg],[lon_std,lat_std],lims
        
        
    def frac_max_filter(self,grid,probs,frac=1./20):
        
        dlon = grid[0][0][1] - grid[0][0][0]
        dlat = grid[1][1][0] - grid[1][0][0]
        
        probmax = np.max(probs)
        
        if probmax == 0:
            return [np.nan,np.nan],[np.nan,np.nan],[None]
        
        select = np.where(probs > frac*probmax)
        
        lonpts = grid[0][select]
        latpts = grid[1][select]
        useprobs = probs[select]
        
        avg = [np.sum(pts*useprobs)/sum(useprobs) for pts in [lonpts,latpts]]
        err = [(max(pts) - min(pts))/2 + dlon for pts in [lonpts,latpts]]
        lims = [[max(pts)+dlon, min(pts)-dlon] for pts in [lonpts,latpts]]
        
        return avg,err,lims
        
    def match(self,grid, c1,c2,c3):
        """
        Combines three probability distributions to isolate points for which the lattitude and longitude 
        values match observation
        
        Inputs
            ------
            c1: ndarray [args,shape=(n,n),dtype=float]
                Normalized distribution as outputted by find_valid.
        
        Outputs:
            stat_array: ndarray [args,shape=(2,2),dtype=float]
                A list consisting in the expected values of the latitude and longitude and the errors.
        """
        
        # We want to compare two functions whose data points are not necessarily the same. Thus, we have to 
        # bin both into a new, global data set.

        comb = c1*c2*c3
                
        #self.dist+=[[grid,comb]]
        
        self.filter = self.frac_max_filter
        avgs,stds,lims = self.filter(grid,comb)
        
        return avgs,stds,lims

    
    def triangulate(self, v1, v2, v3, obs_v2_v1, obs_v3_v2, iterations = 5, obserrs=[None,None], 
                    verbose = True):
        """
        Given v1,v2,v3, equatorial celestial coordinates for three objects, and observed changes in altitude 
        and azimuth of those objects from the ground, returns coordinates of normal vector. i.e. latitude
        and longitude
        
        Input:
            ------
            v1: list [args,shape=(2),dtpe=float]
                The celestial coordinates of the first triangulation object
                
            v2: list [args,shape=(2),dtpe=float]
                The celestial coordinates of the second triangulation object
                
            v3: list [args,shape=(2),dtpe=float]
                The celestial coordinates of the third triangulation object
                
            v2_v1: list [args,shape=(2),dtpe=float]
                The difference in azimuth and altitude between object 2 and object 1
                
            v3_v2: list [args,shape=(2),dtpe=float]
                The difference in azimuth and altitude between object 3 and object 2
        """
        print "Attempting triangulation..."
        
        self.classmod = 1

        if sum(np.array(obserrs)==None) == 0:
            self.obserrs = obserrs
        
        # Calculate difference between v1 and v3
        obs_v3_v1 = [obs_v2_v1[0] + obs_v3_v2[0], obs_v2_v1[1] + obs_v3_v2[1]]
        
        # Set initial lims: whole world
        lims = [[0,360],[-90,90]]
        
        i=0
        while i < iterations:
            if verbose:
                print "Running for lims: " + str(np.round(lims,5).tolist())
            
            # Initiate variable for checking that analysis is succeeding
            success = True
            
            # find the probability distributions for each observation
            grid, c1 = self.find_valid(v1, v2, obs_v2_v1, lims=lims)
            success = np.sum(c1) > 0
            
            # Check success, if true then continue to second observation
            if success:
                _, c2 = self.find_valid(v1, v3, obs_v3_v1, lims=lims)
                success = np.sum(c2) > 0
            
            # Check success, if true then continue to final observation
            if success:
                _, c3 = self.find_valid(v2, v3, obs_v3_v2, lims=lims)
                success = np.sum(c3) > 0
            
            if success:
                av,errs,_lims = self.match(grid,c1,c2,c3)
                success = (np.sum(np.isnan(av)) == 0)*(np.sum(np.isnan(errs)) == 0)
                
                                
            # If failed, try again
            if success == False:
                
                if verbose:
                    print "Error in triangulation. Increasing obserrs in attempt to mend situation."
                self.obserrs = [self.obserrs[j]*10 for j in [0,1]]
                self.classmod *= 2
              
                # i += 1
                
            # Otherwise, calculate average and standard deviation and continue
            else:
                lims = _lims
                
                i+=1

        if True:
            
            self.lon = av[0]
            self.lat = av[1]
            self.errs = errs
            
            conf_95 = [3*e for e in errs]
            
            print "Done."
            return np.array(av), np.array(conf_95)
        
        else:
            print "Unable to triangulate"
            return None
