import struct


def decodeDwCnt(data):
    i,b = struct.unpack('<IB',data)
    dwCnt = (b << 32) + i
    return dwCnt

def decodeSysCnt(data):
    i,h = struct.unpack('<IH',data)
    sysCnt = (h << 32) + i
    return sysCnt

class UwbMsgHeader:
    def __init__(self,header):
        self.panId, = struct.unpack('<H',header[3:5])
        self.dstAddr, = struct.unpack('<H',header[5:7])
        self.srcAddr, = struct.unpack('<H',header[7:9])
        self.siteId, = struct.unpack('<H',header[9:11])
        self.fCode, = struct.unpack('<B',header[11:12])

    def __str__(self):
        return 'srcAddr=%04X.%04X.%04X, fCode=%02X' % (self.siteId, self.panId, self.srcAddr, self.fCode)
    
class UwbMsgPayloadBeacon:
    def __init__(self,payload):
        self.bcnQualityIdx, = struct.unpack('<B',payload[0:1])
        self.bcnTimestampTx = decodeSysCnt(payload[1:7])
        self.bcnPosX, = struct.unpack('<f',payload[7:11])
        self.bcnPosY, = struct.unpack('<f',payload[11:15])
        self.bcnPosZ, = struct.unpack('<f',payload[15:19])
        
    def __str__(self):
        return 'qi=%d, sysCnt=%012X, pos=(%5.2f, %5.2f, %5.2f)' % (self.bcnQualityIdx, self.bcnTimestampTx, self.bcnPosX, self.bcnPosY, self.bcnPosZ)
    
class UwbMsgContent:
    def __init__(self,content):
        self.header = UwbMsgHeader(content[:12])
        bcnMsgFormat = 1
        bcnMsgCode = 0xC
        fCodeBcn = (bcnMsgFormat << 4) | bcnMsgCode
        if self.header.fCode == fCodeBcn: # beacon message
            self.payload = UwbMsgPayloadBeacon(content[12:]) # decode payload
        else:
            self.payload = content[12:] # add raw payload
            
    def __str__(self):
        return '( %s ), ( %s )' % (self.header.__str__(), self.payload.__str__())
    
    def __dict__(self):
        return {"header":self.header.__dict__, "payload":self.payload.__dict__}

class UwbMsgDiagnostics:
    def __init__ (self,diagnostics):
        self.pwrCh, = struct.unpack('<h',diagnostics[0:2])
        self.pwrFP, = struct.unpack('<h',diagnostics[2:4])
        self.nlos,  = struct.unpack('<b',diagnostics[4:5])

    def __str__(self):
        return 'pwrCh=%d, pwrFP=%d, nlos=%d' % (self.pwrCh, self.pwrFP, self.nlos)
    
    def __dict__(self):
        return {"pwrCh":self.pwrCh,"pwrFP": self.pwrFP, "nlos": self.nlos}

class UwbMsgNotification:
    def __init__(self,msg):
        self.msg = msg
        self.timestampRx = decodeDwCnt(msg[:5])
        self.content = UwbMsgContent(msg[5:-5])
        self.diagnostics = UwbMsgDiagnostics(msg[-5:])

    def __str__(self):
        return ' timestampRX: %i [ %s ], [ %s ]' % (self.timestampRx, self.content.__str__() ,self.diagnostics.__str__())
 
    def __dict__(self):
        
        return {"timestampRx" : decodeDwCnt(self.msg[:5]),
                "content": UwbMsgContent(self.msg[5:-5]).__dict__(),
                "diagnosis": UwbMsgDiagnostics(self.msg[-5:]).__dict__()}


if __name__ == '__main__':    
    pass    
