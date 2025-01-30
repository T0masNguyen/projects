import os
import datetime
import pickle
import uwbMessages 
import posMessages


class Importer:

    def __init__(self,logfile,logger=None):
        self.logger = logger

        self.loggerInfo('reading logfile: %s' % logfile)
        fh = open(logfile,'rb')
        self.pklMsgs = []
        try:
            while True:
                self.pklMsgs.extend(pickle.load(fh))
        except EOFError:
            pass
        fh.close()

    def loggerDebug(self,msg):
        if self.logger:
            self.logger.debug(msg)

    def loggerInfo(self,msg):
        if self.logger:
            self.logger.info(msg)

    def hasMsgs(self):
        return len(self.pklMsgs) > 0

    def getMsg(self):
        return self.pklMsgs.pop(0)


    def to_dict(self):
        self.listData = []
        for msg in self.pklMsgs:
            if msg[0] == "uwb":
                self.listData.append({"uwb":uwbMessages.UwbMsgNotification(msg[1]).__dict__(),
                                      "PCtimestamp": msg[2]})
            if msg[0] == "pos":
                self.listData.append({"uwb":posMessages.PosMsgNotification(msg[1]).__dict__(),
                                      "PCtimestamp": msg[2]})

        return self.listData


class Exporter:

    def __init__(self,exportNote=None,logger=None,exportToHomeDir=False):
        self.logger = logger

        if exportToHomeDir:
            self.logPathDir = os.path.expanduser('~') + '/'
        else:
            self.logPathDir = ''
        self.logPathDir += 'pplogs/'

        if not os.path.exists(self.logPathDir):
            os.mkdir(self.logPathDir)

        self.logStart = datetime.datetime.now()
        self.logPathDate = '%s' % self.logStart.strftime('%Y%m%d_%H%M%S')

        self.logPathTail = '_messages'
        if exportNote:
            self.logPathTail += '_%s' % exportNote.replace(' ','_')
        self.logPathTail += '.pkl'

        self.logPath = self.logPathDir + self.logPathDate + self.logPathTail

        self.pklMsgs = []
        self.pklMsgsMaxQueueSize = 1000

    def loggerDebug(self,msg):
        if self.logger:
            self.logger.debug(msg)

    def loggerInfo(self,msg):
        if self.logger:
            self.logger.info(msg)

    def addEntry(self,entry):
        self.pklMsgs.append(entry)

    def addUwbMsg(self,msg):
        #print('add uwb message')
        dti = datetime.datetime.now()
        entry = ('uwb', msg, dti)
        self.addEntry(entry)

    def addPosMsg(self,msg):
        #print('add pos message')
        dti = datetime.datetime.now()
        entry = ('pos', msg, dti)
        self.addEntry(entry)

    def update(self):
        if self.logStart + datetime.timedelta(days=1) < datetime.datetime.now():
            self.exportMsgs()
            self.logStart = datetime.datetime.now()
            self.logPathDate = '%s' % self.logStart.strftime('%Y%m%d_%H%M%S')
            self.logPath = self.logPathDir + self.logPathDate + self.logPathTail

        if len(self.pklMsgs) > self.pklMsgsMaxQueueSize:
            self.exportMsgs()

    def exportMsgs(self):
        self.loggerInfo('exporting collected messages to: %s' % self.logPath)
        fh = open(self.logPath,'ab')
        pickle.dump(self.pklMsgs,fh)
        fh.close()
        self.pklMsgs = []

    def quit(self):
        self.exportMsgs()

    def getName(self):
        return self.logPath
