"""抓 Rapfi 原始报文，看清每一步到底回了什么"""
import queue
import subprocess
import threading
import time

EXE = r"E:\selfevo\rapfi\pbrain-rapfi-windows-avx2.exe"
CWD = r"E:\selfevo\rapfi"

p = subprocess.Popen([EXE], cwd=CWD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1,
                     creationflags=0x08000000)

q: queue.Queue = queue.Queue()


def reader():
    for line in p.stdout:
        q.put(line.rstrip("\r\n"))


threading.Thread(target=reader, daemon=True).start()


def send(cmd):
    print(f">>> {cmd}")
    p.stdin.write(cmd + "\n")
    p.stdin.flush()


def drain(wait=2.0):
    end = time.time() + wait
    got = []
    while time.time() < end:
        try:
            got.append(q.get(timeout=0.2))
        except queue.Empty:
            pass
    for g in got:
        print(f"<<< {g}")
    if not got:
        print("<<< (无输出)")
    return got


send("START 15"); drain(3)
send("INFO timeout_turn 500"); drain(1)
send("INFO timeout_match 900000"); drain(1)
send("INFO max_memory 536870912"); drain(1)
send("BEGIN"); drain(5)
send("TURN 7,7"); drain(5)
send("TURN 8,8"); drain(5)
send("END"); drain(2)
p.terminate()
