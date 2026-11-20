"""Hold a macOS idle-sleep assertion only for the lifetime of an active research job."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path.home()/"research/architecture-queue"
def active_pid():
    try:
        job=json.loads((root/"active.json").read_text())
        if not job:
            return None
        pid=int(job["pid"])
        if pid<=1:
            return None
        cmd=subprocess.run(["/bin/ps","-p",str(pid),"-o","command="],capture_output=True,text=True)
        if cmd.returncode==0 and str(root/"sources"/job["revision"])+"/" in cmd.stdout:
            return pid
    except (FileNotFoundError,KeyError,ValueError):
        pass
    return None

def main():
    while True:
        pid=active_pid()
        if pid:
            subprocess.run(["/usr/bin/caffeinate","-i","-w",str(pid)],check=False)
        time.sleep(1 if pid else 3)
if __name__=="__main__":main()

