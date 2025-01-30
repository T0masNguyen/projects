import wx
from Tools.system_analyzer import *
import cv2 as cv
# ======================================================================================================================
"""
test unit for a gui in wx-python:
-> 4 camera windows
-> 1 3D-diagram for facial Landmarks
-> 1 is in or out of the limit (red/green)
-> 1 display for head pose

--------------------------------------
| cam 1 |                   | red /  |
| cam 2 |      3D-          | green  |
| cam 3 |    diagram        | -------|
| cam 4 |                   | values |
--------------------------------------
|status informations                 |
--------------------------------------         

"""


class ShowCapture(wx.Panel):
    def __init__(self, parent, capture1, capture2, capture3, capture4, fps=15):
        wx.Panel.__init__(self, parent)

        self.capture1 = capture1
        ret1, frame1 = self.capture1.read()
        height, width = frame1.shape[:2]
        parent.SetSize((width * 2 + 20, height * 2 + 35))
        # gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        # findArucoMarkers(frame1, gray)
        frame2 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        self.bmp1 = wx.Bitmap.FromBuffer(width, height, frame1)

        self.capture2 = capture2
        ret2, frame2 = self.capture2.read()
        height, width = frame2.shape[:2]
        #parent.SetSize((width, height))
        frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
        self.bmp2 = wx.Bitmap.FromBuffer(width, height, frame2)

        self.capture3 = capture3
        ret3, frame3 = self.capture3.read()
        height, width = frame3.shape[:2]
        #parent.SetSize((width, height))
        frame3 = cv.cvtColor(frame3, cv.COLOR_BGR2RGB)
        self.bmp3 = wx.Bitmap.FromBuffer(width, height, frame3)

        self.capture4 = capture4
        ret4, frame4 = self.capture4.read()
        height, width = frame4.shape[:2]
        #parent.SetSize((width, height))
        frame4 = cv.cvtColor(frame4, cv.COLOR_BGR2RGB)
        self.bmp4 = wx.Bitmap.FromBuffer(width, height, frame4)


        self.timer = wx.Timer(self)
        self.timer.Start(int(1000./fps))

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_TIMER, self.NextFrame)


    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bmp1,   0,   0)
        dc.DrawBitmap(self.bmp2, 640,   0)
        dc.DrawBitmap(self.bmp3,   0, 480)
        dc.DrawBitmap(self.bmp4, 640, 480)

    def NextFrame(self, event):
        ret1, frame1 = self.capture1.read()
        if ret1:
            frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
            self.bmp1.CopyFromBuffer(frame1)
            self.Refresh()

        ret2, frame2 = self.capture2.read()
        if ret2:
            frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
            self.bmp2.CopyFromBuffer(frame2)
            self.Refresh()

        ret3, frame3 = self.capture3.read()
        if ret3:
            frame3 = cv.cvtColor(frame3, cv.COLOR_BGR2RGB)
            self.bmp3.CopyFromBuffer(frame3)
            self.Refresh()

        ret4, frame4 = self.capture4.read()
        if ret4:
            frame4 = cv.cvtColor(frame4, cv.COLOR_BGR2RGB)
            self.bmp4.CopyFromBuffer(frame4)
            self.Refresh()

# get and filter camlist
camlist = get_cam_names_pygrabber()

filtered_cam_list = []
# Create threads as follows
for idx in range(len(camlist)):
    if (camlist[idx] == 'GENERAL WEBCAM') or (
            camlist[idx] == 'Depstech webcam'):  # filter in get_camera_assignment verschieben
        filtered_cam_list.append(idx)
print(filtered_cam_list)

capture1 = cv.VideoCapture(filtered_cam_list[1], cv.CAP_DSHOW)
capture2 = cv.VideoCapture(filtered_cam_list[2], cv.CAP_DSHOW)
capture3 = cv.VideoCapture(filtered_cam_list[3], cv.CAP_DSHOW)
capture4 = cv.VideoCapture(filtered_cam_list[4], cv.CAP_DSHOW)

app = wx.App()
wx_frame = wx.Frame(None)
cap = ShowCapture(wx_frame, capture1, capture2, capture3, capture4)
wx_frame.Show()
app.MainLoop()
