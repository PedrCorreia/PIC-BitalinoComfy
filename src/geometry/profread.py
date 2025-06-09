import pstats
import io
import sys # Added sys module
import os # Added os module

# Path to your .prof file is now taken from command line
# profile_path = "/home/lugo/ComfyUI/custom_nodes/PIC-BitalinoComfy/src/geometry/worker_profile.prof" # Adjusted path

def main():
    if len(sys.argv) < 2:
        print("Usage: python profread.py <path_to_profile.prof>")
        sys.exit(1)
    
    profile_path = sys.argv[1]
    
    # Determine output file path
    base_name = os.path.splitext(os.path.basename(profile_path))[0]
    output_txt_path = f"{base_name}_readable.txt"

    # Create a string buffer to capture the stats output
    s = io.StringIO()

    try:
        # Load the profiling data
        ps = pstats.Stats(profile_path, stream=s)

        # Sort the statistics. Common sort keys include:
        # 'calls' (call count)
        # 'cumulative' (cumulative time spent in this function and all subfunctions)
        # 'filename'
        # 'ncalls'
        # 'pcalls' (primitive call count)
        # 'line' (line number)
        # 'name' (function name)
        # 'nfl' (name/file/line)
        # 'stdname' (standard name)
        # 'time' (internal time, time spent in the function itself, excluding subfunctions)
        # 'tottime' (same as 'time')
        
        # Sort by cumulative time and print the top 30 functions
        ps.sort_stats(pstats.SortKey.CUMULATIVE)
        s.write(f"Profile: {profile_path}\\n")
        s.write("\\n--- Top 30 functions by CUMULATIVE time ---\\n")
        ps.print_stats(30)

        # Sort by total time (time spent in function itself) and print top 30
        ps.sort_stats(pstats.SortKey.TIME)
        s.write("\\n--- Top 30 functions by TOTAL (internal) time ---\\n")
        ps.print_stats(30)
        
        # You can also print callers and callees if needed, for example:
        # ps.print_callers(30)
        # ps.print_callees(30)

        # Get the captured output
        stats_output = s.getvalue()
        print(stats_output)

        # Optionally, save to a text file
        with open(output_txt_path, "w") as f:
            f.write(stats_output)
        print(f"\\nReadable profile stats also saved to: {output_txt_path}")

    except FileNotFoundError:
        print(f"Error: The profile file was not found at {profile_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        s.close()

if __name__ == "__main__":
    main()
