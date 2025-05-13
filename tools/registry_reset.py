#!/usr/bin/env python
"""
Registry Reset Tool - A utility to reset the PlotRegistry state
"""
import sys
import os
import argparse

# Add the parent directory to the path to allow importing the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.plot.plot_registry import PlotRegistry
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure to run this script from the PIC-2025 directory")
    sys.exit(1)

def reset_registry(show_status=True):
    """Reset the registry and optionally show its status"""
    registry = PlotRegistry.get_instance()
    
    # Capture pre-reset stats
    if show_status:
        pre_signals = len(registry.signals)
        pre_nodes = registry.connected_nodes
    
    # Reset the registry
    registry.reset()
    
    if show_status:
        print(f"Registry reset complete!")
        print(f"Removed {pre_signals} signals and {pre_nodes} node connections")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset the PlotRegistry state")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Don't show status information")
    
    args = parser.parse_args()
    
    # Run the reset
    reset_registry(show_status=not args.quiet)
