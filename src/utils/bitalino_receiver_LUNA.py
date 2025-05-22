import platform
import sys, os
import threading
import time

osDic = {
    "Darwin": f"MacOS/Intel{''.join(platform.python_version().split('.')[:2])}",
    "Linux": "Linux64",
    "Windows": f"Win{platform.architecture()[0][:2]}_{''.join(platform.python_version().split('.')[:2])}",
}
if platform.mac_ver()[0] != "":
    import subprocess
    from os import linesep

    p = subprocess.Popen("sw_vers", stdout=subprocess.PIPE)
    result = p.communicate()[0].decode("utf-8").split(str("\t"))[2].split(linesep)[0]
    if result.startswith("12."):
        print("macOS version is Monterrey!")
        osDic["Darwin"] = "MacOS/Intel310"
        if (
            int(platform.python_version().split(".")[0]) <= 3
            and int(platform.python_version().split(".")[1]) < 10
        ):
            print(f"Python version required is ≥ 3.10. Installed is {platform.python_version()}")
            exit()

abs_file_path = os.path.abspath(__file__)
path_parts = os.path.normpath(abs_file_path).split(os.sep)
base_path = os.path.join(*path_parts[-4:-1])

path_plux = os.path.join(base_path, "PLUX-API-Python3",f"{osDic[platform.system()]}")
#print(f'path_plux {path_plux}')
sys.path.append(path_plux)

import plux

class NewDevice(plux.SignalsDev):
    def __init__(self, address):
        plux.MemoryDev.__init__(address)
        self.time = 0
        self.frequency = 0
        self.last_value = 0

    def onRawFrame(self, nSeq, data):  # onRawFrame takes three arguments
        # print(f'data {data}')
        
        self.last_value = data[2]
        
        if nSeq / self.frequency > self.time:
            return True
        return False


class BitalinoReceiver():
    def __init__(self, bitalino_mac_address, acquisition_duration, sampling_freq, channel_code):
        self.device = None
        
        def bitalinoAcquisition(address, time, freq, code):  # time acquisition for each frequency
            """
            Example acquisition.

            Supported channel number codes:
            {1 channel - 0x01, 2 channels - 0x03, 3 channels - 0x07
            4 channels - 0x0F, 5 channels - 0x1F, 6 channels - 0x3F
            7 channels - 0x7F, 8 channels - 0xFF}

            Maximum acquisition frequencies for number of channels:
            1 channel - 8000, 2 channels - 5000, 3 channels - 4000
            4 channels - 3000, 5 channels - 3000, 6 channels - 2000
            7 channels - 2000, 8 channels - 2000
            """
            self.device = NewDevice(address)
            self.device.time = time  # interval of acquisition
            self.device.frequency = freq
            self.device.start(self.device.frequency, code, 16)
            self.device.loop()  # calls device.onRawFrame until it returns True
            self.device.stop()
            self.device.close()


        def createThreads(address_list, time, freq_list, code_list):
            thread_list = []
            for index in range(len(address_list)):
                thread_list.append(
                    threading.Thread(
                        target=bitalinoAcquisition,
                        args=(
                            address_list[index],
                            time,
                            freq_list[index],
                            code_list[index],
                        ),
                    )
                )
                thread_list[index].start()
            for index in range(len(address_list)):
                thread_list[index].join()
            if platform.system() == "Darwin":
                plux.MacOS.stopMainLoop()    
                
            print('bitalino thread finished')
        
        main_thread = threading.Thread(
            target=createThreads, args=([bitalino_mac_address], acquisition_duration, [sampling_freq], [channel_code])
        )
        main_thread.start()
        
        # if platform.system() == "Darwin":
        #     plux.MacOS.runMainLoop()
        # main_thread.join()
        
    def get_last_value(self):
        if self.device is not None:
            return self.device.last_value
        else:
            return 0

if __name__ == '__main__':
    # bitalino_mac_address = "BTH00:21:08:35:15:32"  # BTH98:D3:C1:FD:30:7F
    bitalino_mac_address = "BTH98:D3:C1:FD:30:7F"  # BTH98:D3:C1:FD:30:7F
    acquisition_duration = 25                                          # seconds
    sampling_freq = 10                               # how many time per second
    channel_code = 0x07
    
    bitalino = BitalinoReceiver(bitalino_mac_address, acquisition_duration, sampling_freq, channel_code)
    
    while True:
        time.sleep(0.5)
        print(f'last value {bitalino.get_last_value()}')
    

    
    
