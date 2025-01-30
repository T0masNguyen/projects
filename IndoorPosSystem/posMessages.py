import struct

class PosMean:
    def __init__(self,mean):
        x,y,z = struct.unpack('<hhh',mean)
        self.x = x / 10. # convert x from dm to m
        self.y = y / 10. # convert y from dm to m
        self.z = z / 10. # convert z from dm to m

    def __str__(self):
        return '( %.2f, %.2f, %.2f)' % (self.x, self.y, self.z)
    
class PosCovariance:
    def __init__(self,cov):
        covxx,covxy,covyy = struct.unpack('<iii',cov)
        self.covxx = covxx / 100. # convert cov (x,x) from dm**2 to m**2
        self.covxy = covxy / 100. # convert cov (x,y) from dm**2 to m**2
        self.covyx = self.covxy   # symmetrical non-diagonal elements
        self.covyy = covyy / 100. # convert cov (y,y) from dm**2 to m**2

    def __str__(self):
        return '( ( %.2f, %.2f ), ( %.2f, %.2f ) )' % (self.covxx, self.covxy, self.covyx, self.covyy)
    
    def __dict__(self):
        return {"xx":self.covxx, "xy":self.covxy, "yx":self.covyx, "yy":self.covyy}

class PosMsgPosition:
    def __init__(self,position):
        self.mean = PosMean(position[0:6])
        self.cov = PosCovariance(position[6:18])

    def __str__(self):
        return 'mean=%s, cov=%s' % (self.mean.__str__(), self.cov.__str__())
    
    def __dict__(self):
        return {'x': self.mean.x,
                'y': self.mean.y,
                'z': self.mean.z, 'cov': self.cov.__dict__()}
    
    
class PosMsgMetaData:
    def __init__(self,meta):
        self.siteId, = struct.unpack('<H',meta)

    def __str__(self):
        return 'siteId=%04X' % self.siteId
    
    def __dict__(self):
        return {'siteId': self.siteId}

    
class PosMsgNotification:
    def __init__(self,msg):
        self.msg = msg

        self.position = PosMsgPosition(msg[0:18])
        self.meta = PosMsgMetaData(msg[18:20])

        
    def __dict__(self):
        
        return {
                "position": PosMsgPosition(self.msg[0:18]).__dict__(),
                "meta": PosMsgMetaData(self.msg[18:20]).__dict__()}

        
    def __str__(self):
        return 'pos: [ %s ], [ %s ]' % (self.position.__str__(), self.meta.__str__())
    
if __name__ == '__main__':    
    pass    
