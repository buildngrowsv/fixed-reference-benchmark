"""Install versioned sources and atomically hand the active queue to the Studio."""
import argparse
import json
from pathlib import Path
import plistlib
import shlex
import subprocess
import tempfile
import uuid

HERE=Path(__file__).resolve().parent
LABEL="com.buildngrowsv.studio-research-queue"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--host",default="genflick-m5ultra")
    args=ap.parse_args()
    def ssh(command,**kwargs):
        return subprocess.run(["ssh","-o","BatchMode=yes",args.host,command],check=True,**kwargs)
    home=ssh("pwd",capture_output=True,text=True).stdout.strip()
    if not home.startswith("/Users/") or "\n" in home:raise ValueError(home)
    root=Path(home)/"research"/"architecture-queue"
    py=Path(home)/"research/routed-expert-splice/.venv/bin/python"
    q=shlex.quote
    queue=json.loads((HERE/"queue.json").read_text())
    ids=[x["id"] for x in queue]
    if len(set(ids))!=len(ids):raise ValueError("duplicate ID")
    for i,j in enumerate(queue):
        if any(p not in ids[:i] for p in j.get("after",{})):raise ValueError("parents must precede children in manifest")
    ssh("mkdir -p "+q(str(root/"sources"))+" "+q(str(root/"runs"))+" "+q(str(Path(home)/"Library/LaunchAgents")))
    inventory = (
        "from pathlib import Path; import sys; "
        "print(*(p.name for p in Path(sys.argv[1]).iterdir() if p.is_dir()), sep='\\n')"
    )
    cached_revisions = set(ssh(
        f"{q(str(py))} -c {q(inventory)} {q(str(root/'sources'))}",
        capture_output=True, text=True,
    ).stdout.splitlines())
    for name,revision in sorted({(j["repo"],j["revision"]) for j in queue}):
        repo=HERE.parents[1]/name
        if not repo.is_dir() or repo.parent!=HERE.parents[1]:raise ValueError(name)
        files=[j["script"] for j in queue if j["revision"]==revision]
        for f in files:subprocess.run(["git","cat-file","-e",revision+":"+f],cwd=repo,check=True)
        if revision in cached_revisions:
            continue
        dest=root/"sources"/revision
        stage=str(dest)+".stage-"+uuid.uuid4().hex
        exists=subprocess.run(["ssh",args.host,"test -d "+q(str(dest))]).returncode
        if exists==0:continue
        if exists!=1:raise RuntimeError("source check failed")
        with tempfile.TemporaryFile() as archive:
            subprocess.run(["git","archive","--format=tar.gz",revision],cwd=repo,stdout=archive,check=True)
            archive.seek(0)
            ssh("mkdir "+q(stage)+" && tar -xzf - -C "+q(stage)+" && mv "+q(stage)+" "+q(str(dest)),stdin=archive)
    def upload(data,path):
        tmp=str(path)+".stage-"+uuid.uuid4().hex
        ssh("cat > "+q(tmp)+" && mv -f "+q(tmp)+" "+q(str(path)),input=data)
    upload((HERE/"worker.py").read_bytes(),root/"worker.py")
    plist=Path(home)/"Library/LaunchAgents"/(LABEL+".plist")
    upload(plistlib.dumps(dict(Label=LABEL,ProgramArguments=[str(py),str(root/"worker.py"),"run"],
              RunAtLoad=True,KeepAlive=True,ThrottleInterval=8,WorkingDirectory=str(root),
              StandardOutPath=str(root/"service.log"),StandardErrorPath=str(root/"service.err"))),plist)
    upload((HERE/"queue.json").read_bytes(),root/"queue.json")
    uid=ssh("id -u",capture_output=True,text=True).stdout.strip()
    ssh(f"launchctl print gui/{uid}/{LABEL} >/dev/null 2>&1 || launchctl bootstrap gui/{uid} "+q(str(plist)))
    awake_label=LABEL+"-awake"
    awake_plist=plist.with_name(awake_label+".plist")
    upload((HERE/"awake.py").read_bytes(),root/"awake.py")
    upload(plistlib.dumps(dict(Label=awake_label,ProgramArguments=[str(py),str(root/"awake.py")],
          RunAtLoad=True,KeepAlive=True,ThrottleInterval=8,StandardErrorPath=str(root/"awake.err"))),awake_plist)
    ssh(f"launchctl print gui/{uid}/{awake_label} >/dev/null 2>&1 || launchctl bootstrap gui/{uid} "+q(str(awake_plist)))
    print(f"Deployed {len(queue)} entries to {args.host}.")
if __name__=="__main__":main()

