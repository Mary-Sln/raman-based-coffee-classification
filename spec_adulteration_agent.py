import os
import sys
import time
import subprocess

BASE_DIR = r"F:\AI-Agents\AI-Kaggle\Raman-IR-AI-Agent"
EXCEL_FILE = os.path.join(BASE_DIR, "coffee_data.xlsx")

# Define steps of the interactive spectroscopy agent pipeline
PIPELINE_STEPS = [
    {"name": "Baseline Correction Agent", "script": "skills/baseline_agent.py"},
    {"name": "Advanced Preprocessing Agent", "script": "skills/preprocessing_agent.py"},
    {"name": "Feature Selection Agent", "script": "skills/feature_selection_agent.py"},
    {"name": "Multi-Model Classification Agent", "script": "skills/classification_agent.py"},
    {"name": "Reporting & Visualization Agent", "script": "skills/reporting_agent.py"}
]

def ask_to_proceed(step_name):
    """Create an interactive pause and ask the user to confirm the next classification phase"""
    while True:
        response = input(f"\n[?] '{step_name}' completed successfully. Proceed to the next step? (y/n): ").strip().lower()
        if response == 'y':
            return True
        elif response == 'n':
            print("\n[-] Pipeline paused by user. Exiting Spec-Adulteration-Agent.")
            return False
        print("[-] Invalid input. Please enter 'y' for yes or 'n' for no.")

def run_sub_agent(agent_name, script_path):
    """Run each sub-agent's script in a separate process and monitor its output"""
    print("\n" + "="*60)
    print(f"[>] Running: {agent_name}...")
    print("="*60)
    
    process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[X] Error: {agent_name} failed with exit code {process.returncode}. Pipeline halted.")
        return False
    return True

def main():
    print("[*] Spec-Adulteration-Agent (Main Orchestrator) Is Active.")
    print(f"[*] Monitoring directory for spectral data file: '{os.path.basename(EXCEL_FILE)}'...")
    
    # Folder monitoring loop for automatic Excel file detection
    while True:
        if os.path.exists(EXCEL_FILE):
            print(f"\n[🚀] Detection Event! Raw data file '{os.path.basename(EXCEL_FILE)}' dropped.")
            print("[*] Initializing automated chemometrics workflow in 2 seconds...")
            time.sleep(2)
            break
        time.sleep(3)
        
    # Chain automation loop of sub-agents with user approval guardrails
    for i, step in enumerate(PIPELINE_STEPS):
        success = run_sub_agent(step["name"], step["script"])
        if not success:
            sys.exit(1)
            
        if i < len(PIPELINE_STEPS) - 1:
            should_continue = ask_to_proceed(step["name"])
            if not should_continue:
                sys.exit(0)

    print("\n" + "="*60)
    print("[🎉] SUCCESS: End-to-End Spectroscopy Adulteration Pipeline Complete!")
    print("="*60)

if __name__ == "__main__":
    main()