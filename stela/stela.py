from transform import Transform
from triangulate import Triangulate

import numpy as np
from astroquery.simbad import Simbad
import urllib2
import os
import serial
import utils
import ephem

from astropy.table import Table, vstack

__PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(__PATH, "data")

class STELA():
    """
    Class that governs the alignment of telescope, star positions, catalogs, and star identification
    
    Quantities and objects:
        STELA.naked: catalog of nearest stars, brightes in sky
        STELA.ard_pos: last stored position of arduino
        STELA.ard_targ: last stored arduino target
    
    Functions:
        STELA.setup_cats:
            Setup catalog files.
        STELA.setup_serial:
            Initiate serial connection with Arduino.
        STELA.get_ref_stars:ls
            Given an estimated latitude and longitude, returns the three brightest stars in the sky to align.
        STELA.gen_mock_obs:
            For testing triangulation
        STELA.triangulate:
            After calling get_ref_stars, and measuring the differences in alt-az coordinates of the three
            points, STELA.triangulate can locate the new latitude and longitude that accounts for the error
            in telescope positioning.
        STELA.set_targ:
            Sends target alt azimuth coordinate to arduino.
        STELA.get_pos:
            Reads current arduino poisition.
        
    """
    
    def __init__(self):
        self.DATA_DIREC = DATA_PATH
        self.simbad = Simbad()
        self.simbad.TIMEOUT = 1000000
        self.simbad.remove_votable_fields('coordinates')
        self.simbad.add_votable_fields('id(NAME)', 'ids','id(NGC)','id(M)','id(HD)','ra','dec',
                                        'otype(V)', 'sp','plx','z_value', 'flux(V)','distance')
        self.reset_cats = False
        self.__triangulation_class = Triangulate()

        self.Transform = Transform()
    
    def setup_cats(self,reset_cats=False):
        """ 
        Sets up the necessary catalogs, prints them to a file. (No parameters)
        
        """
        
        self.__online = self.connect()
        
        if self.__online == False:
            print 'WARNING: No internet connection. Running offline mode.'
        
        print ""

        # Download necessary catalogs
        catlabels = ["GJ","New Galactic Catalog","Messier","Henry Draper"]
        cats = ["gj","ngc","m","hr"]
        catobjs = []

        # If user requests fresh import, do it
        if reset_cats == True and self.__online == True:
            print "Deleting old catalog..."
            for c in cats:
                os.system('rm ' + os.path.join(self.DATA_DIREC, c.lower()+".dat"))
        elif reset_cats == True: 
            print "Requested refresh of catalogs but no internet connection. Find one!"
                
        for i in range(len(cats)):
            fp = os.path.join(self.DATA_DIREC, cats[i].lower() + '.dat')
            
            if os.path.exists(fp) == False:
                if self.__online == False:
                    raise RuntimeError("No saved catalogs, run once with internet")
                    
                print "Downloading " + catlabels[i] +" data..."
                cat = self.simbad.query_catalog(cats[i].upper())
                cat.remove_columns(["Distance_merr","Distance_Q","Distance_perr","Distance_bibcode"])
                select = np.array(cat["RA"] != '') *  np.array(cat["DEC"] != '')
                catobjs += [cat[select]]
                catobjs[-1].write(fp,format='ascii')
                print "Done!"
                
            else:
                print catlabels[i] + " catalog file found."
                catobjs += [Table.read(fp,format='ascii')]

        class cats(): pass
        self.catalogs = cats()
        [self.catalogs.gj, self.catalogs.ngc, self.catalogs.m, self.catalogs.hd] = catobjs

        catobjs = None
        
        # Create catalog of naked eye stars (used in calibration)
        naked_fp = os.path.join(self.DATA_DIREC,'naked.dat')
        if os.path.exists(naked_fp) == False:
            
            print "Setting up naked eye catalogs"
            # remove objects with no recorded magnitude
            select = np.ones(len(self.catalogs.gj),dtype='bool')
            select[np.where(np.isnan(np.array(self.catalogs.gj['FLUX_V'])))[0]] = False
            select[np.where( self.catalogs.gj['FLUX_V'] > 5 )[0]] = False
            naked = self.catalogs.gj[select]
                   
            print len(naked)
            
            self.catalogs.naked = Table(np.unique(naked[:800]))
            print len(self.catalogs.naked)
            self.catalogs.naked.sort("FLUX_V")
            
            # Write it to the catalogs folder
            self.catalogs.naked.write(naked_fp,format='ascii')
        else:
            self.catalogs.naked = Table.read(naked_fp,format='ascii')
        
        print "Done. \n"

    def setup_serial(self):
        """ Searches for a serial connection on the coms. ttyS* ports corresponding to usb COM numbers 
        in /dev/ folder must be fully read/write/ex permissable. If failed, returns a Runtime Error. 
        Some ports just don't work so switching USB ports might solve any problems."""
        
        ports_closed = []
        portsopen=0

        # First, try the expected filepath for a pi
        try:
            self.__ser = serial.Serial("/dev/serial/by-id/usb-Arduino_LLC_Arduino_Micro-if00")
            print "Found the Arduino Micro."
            return
        except:
            pass
        
        # Then, try all of the ttyS* paths (useful on windows)
        for i in range(20):
            # New path to try
            path = "/dev/ttyS" + str(i)
            
            try:
                # Send message, if recieve correct response, keep serial information
                ser = serial.Serial(path)
                ser.write('setup')

                if ser.readline()[:5] == 'STELA':
                    print 'Found STELA arduino running on on COM' + str(i)
                    self.__ser = ser
                    portsopen+=1
                    break
            
            # SerialException could be a sign that permissions need to be changed
            except serial.SerialException as err:
                if err.args[0] == 13:
                    ports_closed += [i]
            
            # IOError means there was no response from the port.
            except IOError:
                pass
          
        # Next, check the pi directory where serial ports are stored
        if os.path.exists("/dev/serial/by-id"):
            for i in os.listdir("/dev/serial/by-id"):
                print "trying other paths..."

                try:
                    ser = serial.Serial("/dev/serial/by-id/" + i)
                    ser.write("setup")
                    if ser.readline()[:5] == "STELA":
                        print "Found connection with: " + i
                        self.__ser = ser
                        portsopen = 1
                        break
                except:
                    pass
                
        # If no serial port, raise error and if permission issues were found.
        if portsopen==0:
            if len(ports_closed) > 0:
                msg = ("Connection Failed. Unable to access ports: " + str(ports_closed) + 
                        ". Try changing permissions.")
            else:
                msg = "Connection Failed. Try different usb port."
                
            raise RuntimeError(msg)

    def search(self, string, list_results=False):
        """
        Search stored catologs for a string. Looks for hits in the IDS column.

        Input
            ------
            string: string
                The search string. Case does not matter, but any misspellings will not fly.

        Optional
            ------
            list_results: bool
                Whether or not to return a list of all hits. If not, returns the best match.

        Output
            ------
            data: dictionary or list of dictionaries
                A dictionary containing all of the information found in the catologs. If 
                list_results == True, a list of dictionaries for each hit.
        """

        print 'Searching for: "' + string + '"\n'
        
        # Check if online or not
        self.__online = self.connect(verbose=False)
        
        matches = Table(self.catalogs.m[0])
        matches.remove_row(0)
        scores = []
        for cat in [self.catalogs.m,self.catalogs.gj,self.catalogs.ngc,self.catalogs.hd]:
            # Load string of all names
            names = cat["IDS"]
            arr = names.view(type=np.recarray)

            # Look for hits in each item of the catalog
            string_low = string.lower()
            low = np.array([s.lower() for s in arr])
            score_l = []

            for l in low:
                allnames = np.array(l.split("|"))
                sc = []
                for a in allnames:
                    if a[:4] == "name":
                        sc += [utils.score(string_low,a[5:])]
                    elif a[:2] == "m " or a[:2] == "hd" or a[:3] == "ngc":
                        sc += [utils.score(string_low,a)]
                    else:
                        sc += [-10.]
                score_l += [max(sc)]

            where = np.where(np.array(score_l) > 0.5)[0]
            if len(where) > 0:
                [matches.add_row(cat[i]) for i in where]
                scores += [score_l[i] for i in where]
        
        if len(matches) > 0:
            ind = scores.index(max(scores))
            result = matches[ind]

        """
        NEED PLANET SEARCHING
        """

        # Check connectivity
        if self.__online == True and type(result) == type(None):
            result = self.simbad.query_object(string)

        if type(result) == type(None):
            return {"Error":"No object found"}

        if list_results == False or len(result) == 1:
            return self.__parse_result(result)
        
        else:
            info = []
            for i in range(len(result)):
                info += [self.__parse_result(result[i])]
        
            return {"list": info}

        
    def __parse_result(self, row):
        """
        Used to turn the catolog entries into dictionaries of information.

        Input
            ------
            row: astropy row or single entry table
                The table to convert.
        
        Output 
            ------
            data: dic [keys=("Name","Otype","ra","dec","Distance","Mag","Sptype","Redshift","Luminosity")
            """
        
        # Attempt to convert a row to a table (tables can be converted to arrays, but rows not)
        as_table = Table(row)
       
        # Extract information to single variables
        [main, name, ids, idngc, idm, 
         idhd, ra, dec, otype, sptype, 
         plx, redshift, mag, dist, distu, 
         distmeth]                       = as_table.as_array()[0]    
        
        # Make a string of all valid names
        usenames = ""
        for i in [name,idm,idngc,idhd]:
            if i != '' and i != 0:
                usenames += i + ", "

        data = {"Name": usenames[:-2]}
        
        # Go through the list, check if variables have information, and add them to dictionary if 
        # information is found
        if mag != None:
            data["Mag"] = "%.3f" % mag
        if redshift > 0.001:
            data["Redshift"] = "%.3f" % redshift
        if otype != None:
            data["Otype"] = otype
        else:
            data["Otype"] = "Unknown"
        if sptype != '':
            data["Sptype"] = str(sptype)

        if plx > 0:                             # Distance calculations from parallax
            data["Plx"] = str(plx)
            distpc = 1000./plx
            distly = distpc*3.2616
            data["Distance"] = "%.2f pc, %.2f ly" % (distpc, distly)

        if redshift != None:   # Distance estimates from redshift
            data["redshift"] = redshift

        if plx > 0 and mag != None:             # Luminosity estimates
            Msun = 4.74
            absmag = mag - 5 *(np.log10(distpc) - 1)
            L = 10 ** (1./2.5 * (Msun - absmag))
            data["Luminosity"] = "%.2f" % L
        
        if dist > 0:
            data["Distance"] = "%.2f %s" % (dist,distu)

        # Add right ascencion and declination information
        data["ra"] = "%s hrs" % ra
        data["dec"] = "%s deg" % dec
        

        # This is for a string with all of the information. Goes down the list, if a key exists, adds 
        # infomation to string. Useful for displaying everything easily.
        order = ["Name", "Otype", "ra", "dec", "Mag", "Distance", "Sptype", "Luminosity", "Redshift"]

        outstring = ''
        for key in order:
            if key in data.keys():
                outstring += key + ": " + data[key] + "\n"

        data["String"] = outstring
        
        return data
            
        
    def set_time(self,datetime):
        """
        Sets the system time.

        Input
            ------
            datetime: string (format)
        """
        l = datetime.split("-")
        date_str = l[1]+l[2]+l[3]+l[4]+l[0]+"."+l[5]
        if os.environ["USER"] == 'pi':
            os.system("sudo date " + date_str)
        else:
            print "Not setting time."

    def set_targ(self, targ):
        """
        Sends the target coordinates in the arduino. 
        
        Input
            ------
            targ: ndarray [args, shape=(2), dtype=float]
                New target coordinates for the arduino.
        """
        
        msg = 'set_targ:' + str(targ)
        self.__ser.write(msg)
        
    def get_pos(self,return_targ=False):
        """ 
        Gets the current arduino position and target. 
        
        Optional
            ------
            return_targ: bool
                If true, returns both the arduino postion and the current arduino target.

        Output
            ------
            ard_pos: ndarray [args, shape=(2), dtype=float]
                The postion sent back from the arduino
        """

        now = ephem.now()
        
        # Send request to arduino through serial, read response
        self.__ser.write('info')
        msg = self.__ser.readline()
        
        # Find alt-az information, break message string into sections
        pos_str = msg[ msg.find('[')+1 : msg.find(']') ]
        targ_str = msg[ msg.find('[',11)+1 : msg.find(']', msg.find(']') + 1) ] 
        
        # Set class objects using parsed string
        self.ard_pos = np.fromstring(pos_str,sep=', ')
        self.ard_targ = np.fromstring(targ_str, sep=', ')
        
        # Save time of update
        self.update_time = now
        
        # Return
        if return_targ==True:
            return self.ard_pos, self.ard_targ
        
        return self.ard_pos
    
    def set_pos(self,newpos):
        """
        Resets the position of the arduino to specified coordinates.
        
        Inputs
            ------
            newpos: ndarray or list [args, shape=(2), dtype=float]
                The coordinates to use to reset the position.
        """
        # create message and send it
        msg = 'set_pos:' + str(newpos)
        self.__ser.write(msg)
        
    def get_ref_stars(self):
        """
        Given an estimation of longitude and latitude, identifies 3 target stars to use as 
        triangulation coordinates
        
        Input:
            ------
            lon_est: float
                The current longitude at which the telescope is set up
            
            lat_est: float
                The current latitude at which the telescope is set up
        
        Optional:
            ------
            representation: string
                The preferred representation of the coordinates ('SkyCoord' or 'String')
        Output:
            ------
            altaz_calib: list [args, shape=(3,2), dtype=float]
                The estimated positions in the altitude azimuth coordinate frame. Can be used to point
                telescope to approximate position of stars.
        """

        [lon_est, lat_est] = self.location

        # Coordinates of all visible stars
        ra= self.catalogs.naked["RA"].data
        dec = self.catalogs.naked["DEC"].data
        
        # One by one, compares catalog entries to the earth normal vector for any that should be 
        # visible in the night sky. After this check, compares entry to other accepted reference stars.
        # If it's too close to others, reject it.
        coors = []
        earthnorm = self.Transform.earthnorm2icrs(self.location)

        for i in range(len( self.catalogs.naked )):

            obj = [utils.hourangle(ra[i]), utils.degree(dec[i])]

            if self.Transform.separation(earthnorm, obj) < 60: 
                # If star is 30 degrees above horizon...

                if sum([self.Transform.separation(c, obj) < 20 for c in coors]) == 0:
                    # Check if star is at least 20 degrees away from the other reference stars...
                    coors += [obj]
                    if len(coors) >= 3:
                        # If we already now have 3 reference stars, we are good.
                        break
        
        # Convert to earth altaz frame
        altaz_calib = [self.Transform.cel2altaz(c, self.location)[0] for c in coors]
        
        # set class objects
        self.altaz_calib = altaz_calib
        self.cel_calib = coors
        
        return altaz_calib
         
    def triangulate(self, v2_v1,v3_v2,iterations=5,verbose=True):
        """
        Used to triangulate the true latitude and longitude corresponding to the norm of the telescope position.
        
        Input
            ------
            v2_v1: list [args,len=2,dtype=float]
                The difference in [azimuth,altitude] between object 2 and object 1
               
            v3_v2: list [args,len=2,dtype=float]
                The difference in [azimuth,anow haveude] between object 3 and object 2
        """
        # Get reference star coordinates
        v = np.array(self.cel_calib)
        
        # Triangulate using the triangulation class (nested in try:except for safety)
        loc, self.loc_errs = self.__triangulation_class.triangulate(v[0],v[1],v[2], v2_v1,v3_v2, 
                                                                iterations=iterations,verbose=verbose)


        zero = self.Transform.earthnorm2icrs([0,0])

        [self.lon,self.lat] = loc - zero
        self.home_coors = [self.lon,self.lat]
        
        self.tel_pos = self.Transform.cel2altaz(v[2],self.home_coors)
        
        try:
            self.set_pos(v[2])
        except:
            pass
        
        self.calibrated = True

        
    def save(self):
        """Saves some of the data to a file."""
        
        [lon,lat] = self.location 
        savedata = {'location': [lon.value,lat.value], 
                    'location_units': [lon.unit.to_string(), lat.unit.to_string()]}
        
        with open(self.savefile,'w') as file:
            file.write(json.dumps(savedata))
            
    def load(self):
        """Loads data from the file."""
        
        with open(self.savefile,'r') as file:
            data = json.loads(file.read())
            
        loc = data['location']
        loc_u = data['location_units']
        
        self.location = [loc[0]*Unit(loc_u[0]),loc[1]*Unit(loc_u[1])]
        
        
    def connect(self,verbose=True):
        """Tests the internet connection.
        
        Optional
            ------
            verbose: bool
                If true, prints out process.
        """ 
        
        if verbose == True:
            print "Testing internet connection..."

        try:
            urllib2.urlopen('http://216.58.192.142', timeout=1)
            if verbose == True:
                print "Connection succeeded!"
            return True
        
        except urllib2.URLError as err:
            if verbose == True:
                print "Connection failed."
            return False
        
        except Exception as e:
            print e
            return False