import platform
import sys
import json
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


sys.path.append("/media/lugo/data/ComfyUI/custom_nodes/bitalino_comfy/src/PLUX-API-Python3/Linux64")

import plux


class NewDevice(plux.SignalsDev):
    def __init__(self, address):
        plux.SignalsDev.__init__(address)  # Directly call the parent class constructor
        self.duration = 0
        self.frequency = 0
        self.data = []  # To store signal data

    def onRawFrame(self, nSeq, data):  # onRawFrame takes three arguments
        print(f"Reading data: {nSeq}, {data}")  # Always print the values being read
        self.data.append({"nSeq": nSeq, "data": data})  # Store data
        return nSeq > self.duration * self.frequency


# example routines


def exampleAcquisition(
    address="BTH20:16:07:18:17:02",
    duration=20,  # Default duration set to 10 seconds
    frequency=1000,
    active_ports=[2],  # Use only 1 port
):  # time acquisition for each frequency
    """
    Example acquisition.
    """
    print("Initializing communications...")
    device = NewDevice(address)
    device.duration = int(duration)  # Duration of acquisition in seconds.
    device.frequency = int(frequency)  # Samples per second.
    
    print("Starting acquisition...")
    start_time = time.time()  # Record the start time
    device.start(device.frequency, active_ports, 16)
    device.loop()  # calls device.onRawFrame until it returns True
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Acquisition completed in {elapsed_time:.2f} seconds.")
    device.stop()
    device.close()
    
    # Save data to JSON file
    output_file = "signal_data.json"
    with open(output_file, "w") as f:
        json.dump(device.data, f)
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    # Use arguments from the terminal (if any) as the first arguments and use the remaining default values.
    exampleAcquisition(*sys.argv[1:])