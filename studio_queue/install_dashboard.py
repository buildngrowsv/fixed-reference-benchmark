"""Install the local loopback-only research dashboard for the current macOS login."""
from pathlib import Path
import os
import plistlib
import shutil
import subprocess
import sys

label="com.buildngrowsv.studio-research-dashboard"
folder=Path.home()/"Library/Application Support/StudioResearchQueue"
folder.mkdir(parents=True,exist_ok=True)
for name in ("dashboard.py","dashboard.html"):
    dest=folder/name
    tmp=dest.with_suffix(".next")
    shutil.copyfile(Path(__file__).with_name(name),tmp);os.replace(tmp,dest)
plist=Path.home()/"Library/LaunchAgents"/(label+".plist")
plist.write_bytes(plistlib.dumps(dict(Label=label,ProgramArguments=[sys.executable,str(folder/"dashboard.py")],
    RunAtLoad=True,KeepAlive=True,ThrottleInterval=10,StandardErrorPath=str(folder/"dashboard.err"))))
domain=f"gui/{os.getuid()}"
if subprocess.run(["launchctl","print",domain+"/"+label],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
    subprocess.run(["launchctl","bootstrap",domain,str(plist)],check=True)
else:
    subprocess.run(["launchctl","kickstart","-k",domain+"/"+label],check=True)

