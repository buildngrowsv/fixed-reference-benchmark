"""Local-only Chrome dashboard. All research status and scheduling live on the Studio."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json
import re
import socket
import subprocess

PORT=8769
ORIGIN=f"http://127.0.0.1:{PORT}"
def remote(action,*ids):
    if action not in ("status","next","hold","resume") or any(not re.fullmatch("[a-z0-9-]{1,100}",s) for s in ids):
        raise ValueError("Invalid queue operation")
    cmd="~/research/routed-expert-splice/.venv/bin/python ~/research/architecture-queue/worker.py "+" ".join((action,*ids))
    r=subprocess.run(["/usr/bin/ssh","-o","BatchMode=yes","-o","ConnectTimeout=4","genflick-m5ultra",cmd],
                     timeout=12,capture_output=True,text=True)
    if r.returncode:raise RuntimeError("Studio unavailable or queue command failed")
    return r.stdout

class Handler(BaseHTTPRequestHandler):
    def respond(self,code,data,kind="application/json"):
        self.send_response(code);self.send_header("Content-Type",kind);self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.headers.get("Host")!=f"127.0.0.1:{PORT}":return self.respond(403,b'{}')
        if self.path=="/":
            return self.respond(200,Path(__file__).with_name("dashboard.html").read_bytes(),"text/html; charset=utf-8")
        if self.path!="/status":return self.respond(404,b'{}')
        try:self.respond(200,remote("status").encode())
        except (RuntimeError,subprocess.TimeoutExpired):self.respond(503,b'{"error":"Studio status unavailable"}')
    def do_POST(self):
        if self.headers.get("Host")!=f"127.0.0.1:{PORT}" or self.headers.get("Origin")!=ORIGIN:return self.respond(403,b'{}')
        if self.path!="/queue" or self.headers.get("Content-Type")!="application/json":return self.respond(400,b'{}')
        try:
            n=int(self.headers.get("Content-Length","0"))
            if not 0<n<512:raise ValueError()
            d=json.loads(self.rfile.read(n));action=d["action"]
            if action=="next":remote(action,d["id"])
            elif action in ("hold","resume"):remote(action)
            else:raise ValueError()
            self.respond(200,b'{"ok":true}')
        except (KeyError,ValueError,TypeError,RuntimeError,subprocess.TimeoutExpired):self.respond(400,b'{"error":"Queue change failed"}')
    def log_message(self,*args):pass
if __name__=="__main__":ThreadingHTTPServer(("127.0.0.1",PORT),Handler).serve_forever()

