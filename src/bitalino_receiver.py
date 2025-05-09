import platform
import sys, os
import threading
import time
from collections import deque

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
print(f'path_plux {path_plux}')
sys.path.append("/media/lugo/data/ComfyUI/custom_nodes/bitalino_comfy/src/PLUX-API-Python3/Linux64")

import plux

class NewDevice(plux.SignalsDev):
    def __init__(self, address):
        #plux.MemoryDev.__init__(address)
        plux.MemoryDev.__init__(address)
        self.time = 0
        self.frequency = 0
        self.last_value0 = 0
        self.last_value1 = 0
        self.last_value2 = 0
        self.last_value3 = 0
        self.last_value4 = 0
        self.last_value5 = 0
        self.ts=0

    def onRawFrame(self, nSeq, data):  # onRawFrame takes three arguments
        self.ts = nSeq / self.frequency
        if len(data) >1:
         self.last_value0 = [self.ts,data[0]]
         if len(data) > 2:
            self.last_value1 = [self.ts,data[1]]
            if len(data) > 3:
                self.last_value2 = [self.ts,data[2]]
                if len(data) > 4:
                    self.last_value3 = [self.ts,data[3]]
                    if len(data) > 5:
                        self.last_value4 = [self.ts,data[4]]
                        if len(data) > 6:
                            self.last_value5 = [self.ts,data[5]]
                
        if nSeq / self.frequency > self.time:
            return True
        return False

class DataCompiler:
    def __init__(self, device, buffer_size, channels):
        """
        Initialize the DataCompiler with deques for the selected channels and a reference to the device.
        :param device: The Bitalino device instance.
        :param buffer_size: The maximum size of each deque.
        :param channels: The number of channels to process (0 to 6).
        """
        self.device = device
        self.buffers = [deque(maxlen=buffer_size) for _ in range(channels)]  # Create deques only for selected channels
        self.last_timestamps = {i: None for i in range(channels)}  # Track the last timestamp for each channel
        self.running = True  # Flag to control the thread

    def compile_data(self):
        """
        Continuously fetch data from the device and store it in the respective deques
        only if the timestamp is new for the channel.
        """
        while self.running:
            if self.device:
                ts = self.device.ts
                data_values = [
                    self.device.last_value0,
                    self.device.last_value1,
                    self.device.last_value2,
                    self.device.last_value3,
                    self.device.last_value4,
                    self.device.last_value5,
                ]
                for i, buffer in enumerate(self.buffers):
                    if data_values[i] and self.last_timestamps[i] != ts:
                        buffer.append((ts, data_values[i][1]))  # Append timestamp and value
                        self.last_timestamps[i] = ts  # Update the last timestamp for the channel

    def stop(self):
        """
        Stop the data compilation process.
        """
        self.running = False

    def get_buffers(self):
        """
        Retrieve the current state of all buffers.
        :return: A list of deques containing the compiled data for the selected channels.
        """
        return self.buffers

class BitalinoReceiver():
    def __init__(self, bitalino_mac_address, acquisition_duration, sampling_freq, channel_code, buffer_size):
        self.device = None
        self.device_initialized = threading.Event()  # Event to signal device initialization
        self.data_compiler = None  # DataCompiler instance

        def bitalinoAcquisition(address, time, freq, code):  # time acquisition for each frequency
            try:
                self.device = NewDevice(address)
                self.device.time = time  # interval of acquisition
                self.device.frequency = freq
                self.device.start(self.device.frequency, 0x3F, 16)  # Pass code_sequence as a list of integers
                self.device_initialized.set()  # Signal that the device is initialized
                self.device.loop()  # calls device.onRawFrame until it returns True
                self.device.stop()
                self.device.close()
            except Exception as e:
                self.device = None
                self.device_initialized.set()  # Signal that initialization failed

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

            # Wait for the acquisition thread to initialize the device
            self.device_initialized.wait()  # Wait until the device is initialized or an error occurs

            if self.device is None:
                return

            # Start the DataCompiler for the selected channels
            self.data_compiler = DataCompiler(self.device, buffer_size, channel_code.bit_length())
            compiler_thread = threading.Thread(target=self.data_compiler.compile_data)
            thread_list.append(compiler_thread)
            compiler_thread.start()

            for thread in thread_list:
                thread.join()

            if platform.system() == "Darwin":
                plux.MacOS.stopMainLoop()

        main_thread = threading.Thread(
            target=createThreads, args=([bitalino_mac_address], acquisition_duration, [sampling_freq], [channel_code])
        )
        main_thread.start()

    def stop(self):
        """
        Stop the data compilation process and the acquisition.
        """
        if self.data_compiler:
            self.data_compiler.stop()

    def get_buffers(self):
        """
        Retrieve the current state of all buffers.
        :return: A list of deques containing the compiled data for the selected channels.
        """
        if self.data_compiler:
            return self.data_compiler.get_buffers()
        return []

    def get_last_value(self):
        if self.device is not None:
            return [self.device.ts, self.device.last_value0, self.device.last_value1, self.device.last_value2, 
                    self.device.last_value3, self.device.last_value4, self.device.last_value5]
        else:
            return []

if __name__ == '__main__':
    bitalino_mac_address = "BTH20:16:07:18:17:02"  # BTH98:D3:C1:FD:30:7F
    acquisition_duration = 60                                         # seconds
    sampling_freq = 100                               # how many time per second
    channel_code = 0x7F
    buffer_size = 1000  # Maximum size of each deque
    
    bitalino = BitalinoReceiver(bitalino_mac_address, acquisition_duration, sampling_freq, channel_code, buffer_size)
    
    while True:
        time.sleep(0.5)
        print("sample")
        print(f'last value {bitalino.get_last_value()}')
        print(f'buffers {bitalino.get_buffers()}')




