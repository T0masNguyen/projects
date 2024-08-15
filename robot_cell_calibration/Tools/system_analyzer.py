from pygrabber.dshow_graph import FilterGraph
import platform
import cv2 as cv
import winreg
import wmi  # --------------------------------------------------------------------- auslesen der aktuell aktiven Kameras

"""
unit for analyzing the system for windows
-> not platform independent!
-> which Backend is for Linux useful?
"""


def check_sys_config(verbose=False):
    """
    function to detect system features
    :param verbose:
    :return:
    """
    if verbose:
        print('------------------------------------------------------------------------------ Get system configuration')
    # -------------------------------------------------------------------------------------
    os = platform.system()
    pc_name = platform.node()
    os_version = platform.version()
    cpu = platform.processor()
    machine_type = platform.machine()
    python_version = platform.python_version()
    # --------------------------------------------------------------------------------------
    ocv_version = cv.version.opencv_version
    gpu_count = cv.cuda.getCudaEnabledDeviceCount()
    if gpu_count != 0:
        print(cv.cuda_DeviceInfo.deviceID(0))           # all GPU's or NVIDIA only?
    cpu_count = cv.getNumberOfCPUs()
    thread_count = cv.getNumThreads()
    cuda_devices = cv.cuda.getCudaEnabledDeviceCount()
    # cv.cuda_DeviceInfo.majorVersion(cv.cuda_DeviceInfo.deviceID()) )#, cv.cuda_DeviceInfo.deviceID() )
    # -------------------------------------------------- create iterable output-dictionary
    sys_config = {'hostname': pc_name,
                  'system': {'os': os, 'os version': os_version},
                  'cpu': cpu, 'machine type': machine_type,
                  'CPU count': cpu_count, 'thread count': thread_count, 'cuda devices': cuda_devices,
                  'python version': python_version,
                  'opencv version': ocv_version}
    if verbose:
        print(f'-> System configuration:\n'
              f'       Hostname:       {pc_name}\n'
              f'       System:         {os} - {os_version}\n'
              f'       CPU info:       {cpu} - type: {machine_type}\n'
              f'       CPU count:      {cpu_count} - threads: {thread_count}\n'
              f'       cuda devices:   {cuda_devices}\n'
              f'       Python version: {python_version}\n'
              f'       OpenCV version: {ocv_version}')

    return sys_config

def check_cam_structure_winreg():
    # https://stackoverflow.com/questions/41298588/opencv-videocapture-device-index-device-number
    # ---------------------------------------------------------------- scan the Windows registry for all usb-entries
    # all connected USB devices
    usb_devices = []
    index = 0
    # registry: Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\USB\VID_1B3F&PID_1167\5&f65b9aa&0&1
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Enum') as root:
        with winreg.OpenKey(root, 'USB') as root_usb:
            while True:
                try:
                    subkey = winreg.EnumKey(root_usb, index)
                    usb_devices.append(subkey)
                    index += 1
                except WindowsError as e:
                    if e.winerror == 259:  # No more data is available
                        break
                    elif e.winerror == 234:  # more data is available
                        index += 1
                        continue
                    raise e
    # --------------------------------------------------------------------------------------------------------------
    usb_sub_devices = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Enum') as root:
        for n in usb_devices:
            name = ('USB' + '\\' + n)
            with winreg.OpenKey(root, name) as root_usb:
                index = 0
                while True:
                    try:
                        subkey = winreg.EnumKey(root_usb, index)
                        usb_sub_devices.append((name + '\\' + subkey))
                        index += 1
                    except WindowsError as e:
                        if e.winerror == 259:  # No more data is available
                            break
                        elif e.winerror == 234:  # more data is available
                            index += 1
                            continue
                        raise e
    # --------------------------------------------------------------------------------------------------------------
    usb_video_devices = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Enum') as root:
        for n in usb_sub_devices:
            name = (n)
            with winreg.OpenKey(root, name) as root_usb:
                index = 0
                vid = ''
                friendly_name = ''
                location = ''
                ContainerID = ''
                while True:
                    try:
                        subname, subvalue, type = winreg.EnumValue(root_usb, index)
                        if 'usbvideo' == subvalue:
                            vid = 'usbvideo'
                        if 'FriendlyName' == subname:
                            friendlyname = subvalue
                        if 'LocationInformation' == subname:
                            location = subvalue
                        if 'ContainerID' == subname:
                            ContainerID = subvalue

                        index += 1
                    except WindowsError as e:
                        if e.winerror == 259:  # No more data is available
                            break
                        elif e.winerror == 234:  # more data is available
                            index += 1
                            continue
                        raise e
                if vid != '':
                    usb_video_devices.append((name + '\\' + subname))
                    # print(name, ' -> ', location, ' -> ', friendlyname, ' -> ', ContainerID)


    # https://stackoverflow.com/questions/60130625/associate-usb-video-capture-device-friendly-name-with-opencv-port-number-in-pyth
    # https://pypi.org/project/WMI/
    print('------------------------------------------------------------------------------------------- via WMI')
    c = wmi.WMI()
    wql = "Select * From Win32_USBControllerDevice"
    for item in c.query(wql):
        a = item.Dependent.PNPClass
        b = item.Dependent.Name.upper()
        if (a != None) and (b != None):
            if (a.upper() == 'CAMERA') and 'AUDIO' not in b:
                print(item.Dependent.DeviceID, item.Dependent.Service, item.Dependent.HardwareID, item.Dependent.Name, item.Dependent.Present, item.Dependent.Status)

def get_cam_names_pygrabber():
    """
    # https://github.com/bunkahle/pygrabber -> using DirectShow!
    :return: DShow filter graph
    """
    graph = FilterGraph()
    print('--------------------------------------------------------------------------- via pygrabber and DSHOW')
    print('detected camera streams: ' + '\t' + str(graph.get_input_devices()))  # list of camera device

    return graph.get_input_devices()


# ======================================================================================================================
if __name__ == "__main__":
    check_sys_config(verbose=True)
    check_cam_structure_winreg()
    get_cam_names_pygrabber()
