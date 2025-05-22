import time
import numpy as np
import matplotlib.pyplot as plt
from comfy.Registry.comfy_signal_generator_node import ComfySignalGeneratorNode
from comfy.phy.ecg import ECGNode
from comfy.plot.comfy_plot_registry_node import ComfyPlotRegistryNode
from src.registry.signal_registry import SignalRegistry

def main():
    # --- Start signal generator node ---
    gen_node = ComfySignalGeneratorNode()
    signal_id = "GEN_ECG"
    gen_node.generate(
        signal_type="ecg_waveform",
        sampling_freq=1000,
        freq=1.2,
        signal_id=signal_id,
        duration_sec=12
    )

    # --- Wait for at least 10 samples to be available ---
    registry = SignalRegistry.get_instance()
    print("Waiting for generated signal...")
    while True:
        sig = registry.get_signal(signal_id)
        if sig and 't' in sig and len(sig['t']) >= 10:
            break
        time.sleep(0.01)

    # --- Start ECG processing node ---
    ecg_node = ECGNode()
    processed_signal_id = "ECG_PROCESSED"
    ecg_node.process_ecg(signal_id, show_peaks=True, output_signal_id=processed_signal_id)

    # --- Start plot registry node (UI) ---
    plot_node = ComfyPlotRegistryNode()
    plot_node.plot()  # This will open the visualization UI in a background thread

    # --- Let the system run for the duration of the signal ---
    print("Running for 14 seconds to allow processing and visualization...")
    time.sleep(14)

    # Optionally, you can fetch and plot the signals statically as before
    raw = registry.get_signal(signal_id)
    processed = registry.get_signal(processed_signal_id)
    plt.figure(figsize=(14, 6))
    plt.plot(raw['t'], raw['v'], label='Raw ECG', alpha=0.5)
    if processed and 't' in processed and 'v' in processed:
        plt.plot(processed['t'], processed['v'], label='Processed ECG', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('ECG: Raw vs Processed')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
