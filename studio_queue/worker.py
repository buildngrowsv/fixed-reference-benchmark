"""Serial, durable dispatcher for the Mac Studio research queue."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HOME = Path(os.environ.get("STUDIO_RESEARCH_QUEUE", Path.home()/"research"/"architecture-queue"))
MODEL_CACHE = Path.home() / "research" / "routed-expert-splice" / "cache" / "large-opener"
PYTHON = Path(os.environ.get("STUDIO_RESEARCH_PYTHON", Path.home()/"research"/"routed-expert-splice"/".venv"/"bin"/"python"))

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default

def put(path, data):
    temp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data,indent=2)+"\n")
    os.replace(temp,path)

def event(kind,job,**data):
    with (HOME/"events.jsonl").open("a") as f:
        f.write(json.dumps(dict(time=now(),event=kind,job=job,**data))+"\n")

def directory(job):
    return HOME/"runs"/job["id"]

def load_queue():
    jobs=read(HOME/"queue.json",[])
    seen=set()
    for j in jobs:
        name=j["id"]
        if not re.fullmatch("[a-z0-9-]+",name) or name in seen:raise ValueError(f"invalid job ID: {name}")
        if not re.fullmatch("[0-9a-f]{40}",j["revision"]):raise ValueError(f"unversioned source: {name}")
        if Path(j["script"]).is_absolute() or ".." in Path(j["script"]).parts:raise ValueError(name)
        if any(not re.fullmatch("[0-9a-f]{40}", revision) for revision in j.get("model_snapshots", [])):
            raise ValueError(f"invalid model snapshot revision: {name}")
        seen.add(name)
    if any(p not in seen for j in jobs for p in j.get("after",{})):raise ValueError("missing queue dependency")
    return jobs

def evaluate(job):
    run=directory(job)
    receipt=read(run/"receipt.json",{})
    if receipt.get("status") in ("done","failed"):return receipt["status"]
    for parent,condition in job.get("after",{}).items():
        p=HOME/"runs"/parent
        r=read(p/"receipt.json",{})
        if r.get("status")=="failed":return "blocked"
        if r.get("status")!="done":return "waiting"
        if condition!="any" and read(p/"result.json",{}).get("decision")!=condition:return "skipped"
    if any(not (MODEL_CACHE / r / ".downloaded").is_file() for r in job.get("model_snapshots", [])):
        return "waiting"
    return "ready"

def status():
    jobs=load_queue()
    parents={j["id"]:evaluate(j) for j in jobs}
    for _ in jobs:
        for j in jobs:
            if parents[j["id"]]=="waiting" and any(parents[p]=="skipped" for p in j.get("after",{})):
                parents[j["id"]]="skipped"
    ordering=read(HOME/"priority.json",[])
    jobs.sort(key=lambda j:ordering.index(j["id"]) if j["id"] in ordering else len(ordering))
    return jobs,parents

def work(job,lease):
    name=job["id"];run=directory(job);run.mkdir(parents=True,exist_ok=True)
    source=HOME/"sources"/job["revision"]
    previous=read(run/"receipt.json",{})
    if previous.get("revision",job["revision"])!=job["revision"]:raise ValueError(f"job ID reused with new source: {name}")
    attempt=previous.get("attempt",0)+1
    record=dict(job=name,revision=job["revision"],attempt=attempt,started=now(),status="running")
    env={**os.environ,"STUDIO_JOB_DIR":str(run),"STUDIO_QUEUE_ROOT":str(HOME),"PYTHONUNBUFFERED":"1"}
    put(run/"receipt.json",record)
    event("started",name,attempt=attempt,revision=job["revision"])
    exitcode=1
    try:
        with (run/"run.log").open("a") as log:
            child=subprocess.Popen([str(PYTHON),str(source/job["script"]),*job.get("args",[])],
                                   cwd=source,stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=(lease,))
            put(HOME/"active.json",dict(job=name,pid=child.pid,started=record["started"],revision=job["revision"]))
            exitcode=child.wait()
        result=read(run/"result.json",{}) if exitcode==0 else {}
        succeeded=bool(result.get("decision"))
    except (OSError,ValueError) as exc:
        succeeded=False;record["error"]=str(exc)
    state="done" if succeeded else ("retry" if attempt<2 else "failed")
    record.update(status=state,finished=now(),exit=exitcode)
    put(run/"receipt.json",record);put(HOME/"active.json",None)
    event(state,name,exit=exitcode,decision=result["decision"] if succeeded else None)

def dispatch(lease):
    # A child inherits the lock. A supervisor restart cannot launch a second GPU job.
    active=read(HOME/"active.json")
    if active:
        run=HOME/"runs"/active["job"];r=read(run/"receipt.json",{})
        if r.get("status")=="running":
            result=read(run/"result.json",{})
            r.update(status="done" if result.get("decision") else ("retry" if r.get("attempt",0)<2 else "failed"),recovered=now())
            put(run/"receipt.json",r);event("recovered",active["job"],status=r["status"])
        put(HOME/"active.json",None)
    while True:
        if not (HOME/"HOLD").exists():
            jobs,states=status()
            job=next((j for j in jobs if states[j["id"]]=="ready"),None)
            if job:
                work(job,lease)
                continue
        time.sleep(3)

def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=("run","status","front","next","hold","resume"));p.add_argument("ids",nargs="*");a=p.parse_args()
    HOME.mkdir(parents=True,exist_ok=True)
    if a.action=="run":
        with (HOME/"worker.lock").open("a+") as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return
            dispatch(lock.fileno())
    elif a.action in ("front","next"):
        jobs,_=status()
        if len(a.ids)!=len(set(a.ids)) or set(a.ids)-{j["id"] for j in jobs}:p.error("unknown or duplicate job ID")
        priority=a.ids
        if a.action=="next":
            priority+= [j["id"] for j in jobs if j["id"] not in priority]
        put(HOME/"priority.json",priority)
    elif a.action=="hold":(HOME/"HOLD").touch()
    elif a.action=="resume":(HOME/"HOLD").unlink(missing_ok=True)
    else:
        jobs,states=status();active=read(HOME/"active.json")
        if active:states[active["job"]]="running"
        items=[]
        for j in jobs:
            path=directory(j)
            items.append(dict(id=j["id"],status=states[j["id"]],repo=j["repo"],after=j.get("after",{}),
                              receipt=read(path/"receipt.json"),progress=read(path/"progress.json"),
                              result=read(path/"result.json")))
        print(json.dumps(dict(held=(HOME/"HOLD").exists(),active=active,jobs=items),indent=2))
if __name__=="__main__":main()

