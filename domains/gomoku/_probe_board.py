"""试 BOARD 的几种可能格式，找出 Rapfi 认的那一种"""
import subprocess, time, queue, threading

EXE = "rapfi/pbrain-rapfi-windows-avx2.exe"


class Eng:
    def __init__(self):
        self.p = subprocess.Popen([EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  cwd="rapfi")
        self.q = queue.Queue()
        threading.Thread(target=self._r, daemon=True).start()

    def _r(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip())
        self.q.put(None)

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def drain(self, t=1.2):
        end = time.time() + t
        got = []
        while time.time() < end:
            try:
                v = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if v is None:
                break
            got.append(v)
        return got

    def close(self):
        try:
            self.send("END")
            self.p.wait(timeout=3)
        except Exception:
            self.p.kill()


CASES = {
    "A 顺序+who(1/2)": "BOARD\n15,15,5\n7,7,1\n5,7,2\n5,6,1\n7,6,2\nDONE",
    "B 顺序+who(0/1)": "BOARD\n15,15,5\n7,7,0\n5,7,1\n5,6,0\n7,6,1\nDONE",
    "C y,x 顺序": "BOARD\n15,15,5\n7,7,1\n7,5,2\n6,5,1\n6,7,2\nDONE",
    "D 无who": "BOARD\n15,15,5\n7,7\n5,7\n5,6\n7,6\nDONE",
    "E who=1/2 但黑白反": "BOARD\n15,15,5\n7,7,2\n5,7,1\n5,6,2\n7,6,1\nDONE",
    "F 空盘": "BOARD\n15,15,5\nDONE",
}

for name, payload in CASES.items():
    e = Eng()
    e.send("START 15")
    e.drain(0.6)
    e.send("INFO timeout_turn 300")
    e.send("INFO strength 100")
    e.send(payload)
    out = e.drain(1.5)
    print(f"{name:24s} -> {[x for x in out if not x.upper().startswith(('MESSAGE','INFO'))][:4]}")
    e.close()
