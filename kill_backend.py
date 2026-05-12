import subprocess, sys, time

def pids_on_port(port):
    pids = []
    out = subprocess.check_output(["netstat", "-ano"], text=True, errors="replace", stderr=subprocess.DEVNULL)
    for line in out.splitlines():
        if f":{port} " in line and "LISTENING" in line:
            try: pids.append(int(line.strip().split()[-1]))
            except: pass
    return list(set(pids))

port = 8004
pids = pids_on_port(port)
if not pids:
    print(f"Port {port} is already free.")
    sys.exit(0)
print(f"Killing PIDs on port {port}: {pids}")
for p in pids:
    subprocess.call(["taskkill", "/F", "/T", "/PID", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)
remaining = pids_on_port(port)
if remaining:
    print(f"Still running: {remaining}")
    print("Try running this script as Administrator.")
else:
    print(f"Port {port} is now free.")
