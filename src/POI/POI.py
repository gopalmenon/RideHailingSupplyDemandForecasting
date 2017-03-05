'''
Created on Feb 16, 2017

@author: Kimberly Williamson
'''

def loopPOIClasses(pClass):
    f = pClass.split(':')
    fac = f[1].strip()
    if '#' in f[0]:
        first = f[0].split('#')[0]
        second = f[0].split('#')[1]
        entry = POIClass(fac, first, second)
    else:
        first = f[0]
        entry = POIClass(fac, first)
    
    return entry

class POIDivided:
    '''
    Store POIDivided data
    '''
    def __init__(self, poiClass):
        self.poiClass = poiClass

    def separatePOI(self):
        poiSplit = self.poiClass.split('\t')
        classList = []
        for pClass in poiSplit:
            #loop through each pClass to separate out the facilities from the class levels.
            entry = loopPOIClasses(pClass)
            classList.append(entry.createCatagories())
        #Return a tuple of the hash with a list of the POI classes
        return classList
        
        
class POIClass:
    '''
    Store POI Classes
    '''
    def __init__(self, facilities, firstLevel, secondLevel=None):
        self.facilities = facilities
        self.firstLevel = firstLevel
        self.secondLevel = secondLevel
    
    def getNumberFacilities(self):
        return self.facilities
    
    def createCatagories(self):
        if self.secondLevel is None:
            self.secondLevel = ''
        return self.firstLevel +'#'+ self.secondLevel
    
class POICatagoryAssignment:
    def __init__(self, poiCatagories, districtHash,poiClass, districtList):
        self.poiCatagories = poiCatagories
        self.districtHash = districtHash
        self.poiClass = poiClass
        self.districtList = districtList
        
    def createPOIDict(self):
        poiSplit = self.poiClass.split('\t')
        for pClass in poiSplit:
            entry = loopPOIClasses(pClass)
            poiCategory = entry.createCatagories()
            idx = self.poiCatagories.index(poiCategory)
            #need to find out how to replace a value
            del self.districtList[idx]
            self.districtList.insert(idx, entry.getNumberFacilities())
        return self.districtHash, self.districtList
        
            
class ReadPOI:
    '''
    Read the POI from file
    '''
    def __init__(self, file):
        self.file = file
        
    def readFile(self):
        poiDict = {}
        poiCategories = set([])
        poiListCategories = []
        #Read the POI file
        with open(self.file, 'r') as poiFile:
            #Loop through the each line and separate the district hash from the POI class.  Then add POIDivided results to a dictionary.
            for row in poiFile:
                rowSplit = row.split('\t',1)
                poiList = POIDivided(rowSplit[1]).separatePOI()
                for item in poiList:
                    poiCategories.add(item)
            poiListCategories = list(poiCategories)
            totalCatagories = len(poiListCategories)
        with open(self.file, 'r') as poiFile:
            for row in poiFile:
                districtList = []
                for z in range(totalCatagories):
                    districtList.append(0)
                rowSplit = row.split('\t',1)
                x,v = POICatagoryAssignment(poiListCategories, rowSplit[0], rowSplit[1], districtList).createPOIDict()
                poiDict[x] = v
        return poiDict

            


        