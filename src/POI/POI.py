'''
Created on Feb 16, 2017

@author: Kimberly Williamson
'''

class POIDivided:
    '''
    Store POIDivided data
    '''
    def __init__(self, districtHash, poiClass):
        self.districtHash = districtHash
        self.poiClass = poiClass
        
    def separatePOI(self):
        #Create empty list to store individual POI classes
        poiList = []
        poiSplit = self.poiClass.split('\t')
        #loop through each pClass to separate out the facilities from the class levels.
        for pClass in poiSplit:
            f = pClass.split(':')
            fac = f[1].strip()
            if '#' in f[0]:
                first = f[0].split('#')[0]
                second = f[0].split('#')[1]
                entry = POIClass(fac, first, second)
            else:
                first = f[0]
                entry =POIClass(fac, first)
            poiList.append(entry.createList())
        #Return a tuple of the hash with a list of the POI classes
        return self.districtHash, poiList
        
class POIClass:
    '''
    Store POI Classes
    '''
    def __init__(self, facilities, firstLevel, secondLevel=None):
        self.facilities = facilities
        self.firstLevel = firstLevel
        self.secondLevel = secondLevel
    
    def createList(self):
        return [self.facilities,self.firstLevel,self.secondLevel]
        
class ReadPOI:
    '''
    Read the POI from file
    '''
    def __init__(self, file):
        self.file = file
        
    def readFile(self):
        poiDict = {}
        #Read the POI file
        with open(self.file, 'r') as poiFile:
            #Loop through the each line and separate the district hash from the POI class.  Then add POIDivided results to a dictionary.
            for row in poiFile:
                rowSplit = row.split('\t',1)
                x,v = POIDivided(rowSplit[0],rowSplit[1]).separatePOI()
                poiDict[x] = v
        return poiDict
            


        