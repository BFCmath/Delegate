#!/usr/bin/env python3
import os
import subprocess
import shlex  # For safely splitting string arguments
from dotenv import load_dotenv

def main():
    """
    Main function to run the benchmark tests, converted from the original bash script.
    """
    # --- Configuration ---
    load_dotenv()
    # Target model name list
    target_models = ["minddeepsearch"]

    # Common parameters for both RACE and Citation evaluations
    raw_data_dir = "data/test_data/raw_data"
    output_dir = "results"
    n_total_process = 1
    query_data_path = "data/prompt_data/query.jsonl"

    # --- Optional Parameters ---
    # To enable an option, set it to its string value (e.g., "--limit 10").
    # To disable, set it to None or an empty string.

    # Limit on number of prompts to process (for testing)
    limit = "--limit 1"
    # limit = None

    # Skip article cleaning step
    # skip_cleaning = None
    skip_cleaning = "--skip_cleaning"

    # Only process specific language data (choose one or neither)
    only_zh = None       # "--only_zh"
    only_en = "--only_en"  # "--only_en"

    # Force re-evaluation even if results exist
    force = None
    # force = "--force"

    # Specify log output file
    output_log_file = "output.log"

    # --- Script Execution ---

    # Helper function to print to console and append to log file (like tee -a)
    def log_and_print(message, file_handle):
        print(message)
        file_handle.write(message + '\n')
        file_handle.flush() # Ensure message is written immediately

    # Clear log file by opening in write mode
    # FIX: Added encoding='utf-8' to handle non-ASCII characters in logs.
    with open(output_log_file, 'w', encoding='utf-8') as f:
        f.write(f"Starting benchmark tests, log output to: {output_log_file}\n")

    # Open log file in append mode for the rest of the script
    # FIX: Added encoding='utf-8' here as well.
    with open(output_log_file, 'a', encoding='utf-8') as log_file:
        # Loop through each model in the target models list
        for target_model in target_models:
            print(f"Running benchmark for target model: {target_model}")
            log_file.write(f"\n\n========== Starting evaluation for {target_model} ==========\n\n")

            # --- Phase 1: RACE Evaluation ---
            phase_header = f"==== Phase 1: Running RACE Evaluation for {target_model} ===="
            log_and_print(phase_header, log_file)
            
            # Create the model-specific output directory
            race_output = os.path.join(output_dir, "race", target_model)
            os.makedirs(race_output, exist_ok=True)

            # Base command for current target model as a list
            cmd_list = [
                "python",
                "-u",  # Unbuffered output, same as the original script
                "deepresearch_bench_race.py",
                target_model,
                "--raw_data_dir", raw_data_dir,
                "--max_workers", str(n_total_process),
                "--query_file", query_data_path,
                "--output_dir", race_output,
            ]

            # Add optional parameters if they are set
            if limit:
                # Use shlex.split for arguments with values like "--limit 10"
                cmd_list.extend(shlex.split(limit))
            if skip_cleaning:
                cmd_list.append(skip_cleaning)
            if only_zh:
                cmd_list.append(only_zh)
            if only_en:
                cmd_list.append(only_en)
            if force:
                cmd_list.append(force)

            # Execute command and append stdout and stderr to the log file
            # Using shlex.join for a safe, readable representation of the command
            exec_msg = f"Executing command: {shlex.join(cmd_list)}"
            log_and_print(exec_msg, log_file)
            
            try:
                # subprocess.run waits for the command to complete
                # stdout and stderr are redirected to the log file handle
                subprocess.run(
                    cmd_list,
                    stdout=log_file,
                    stderr=subprocess.STDOUT, # Redirect stderr to the same place as stdout
                    check=True, # Raise an exception if the command returns a non-zero exit code
                    text=True,   # Write output as text
                    encoding='utf-8' # FIX: Explicitly set encoding for the subprocess output.
                )
            except FileNotFoundError:
                error_msg = f"Error: The command 'python' was not found. Make sure Python is in your PATH."
                log_and_print(error_msg, log_file)
            except subprocess.CalledProcessError as e:
                error_msg = f"Command failed with exit code {e.returncode}."
                log_and_print(error_msg, log_file)
            
            print(f"Completed RACE benchmark test for target model: {target_model}")
            log_file.write(f"\n========== RACE test completed for {target_model} ==========\n")
            print("--------------------------------------------------")

    print(f"All benchmark tests completed. Logs saved in {output_log_file}")


if __name__ == "__main__":
    main()