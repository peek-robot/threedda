import os
import time

def get_slurm_time_left(verbose=0):
    start_time = os.getenv("SLURM_JOB_START_TIME")
    end_time = os.getenv("SLURM_JOB_END_TIME")

    if start_time and end_time:
        start_time = int(start_time)
        end_time = int(end_time)
        
        total_time = end_time - start_time  # Time limit in seconds
        remaining_time = end_time - time.time()  # Time left in seconds

        if remaining_time > 0:
            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            seconds = int(remaining_time % 60)
            if verbose:
                print(f"SLURM Job has {hours:02}:{minutes:02}:{seconds:02} left.")
            return remaining_time
        elif verbose:
            print("SLURM Job has exceeded its time limit!")
    elif verbose:
        print("Could not determine SLURM job time limit.")
    return None