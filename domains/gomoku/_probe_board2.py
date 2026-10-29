"""看清 BOARD 的原始往来：每发一步，把引擎吐的每一行都打出来"""
import subprocess, time, queue, threading

EXE = "E:/selfevo/rapfi/pbrain-rapfi-windows-avx2.exe"
CWD = "E:/selfevo/rapfi"


def run(payload, label):
    p = subprocess.Popen([EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=CWD)
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(l.rstrip()) for l in p.stdout] or q.put(None),
                     daemon=True).start()

    def drain(t=1.0):
        end, got = time.time() + t, []
        while time.time() < end:
            try:
                v = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if v is None:
                break
            got.append(v)
        return got

    print(f"\n===== {label} =====")
    p.stdin.write("START 15\n"); p.stdin.flush()
    print("START  ->", drain(0.8))
    p.stdin.write("INFO timeout_turn 300\nINFO strength 100\n"); p.stdin.flush()
    time.sleep(0.3); drain(0.4)
    print("BOARD  ->")
    for line in payload.split("\n"):
        print("   >>", line)
    p.stdin.write(payload + "\n"); p.stdin.flush()
    print("   <<", drain(2.0))
    try:
        p.stdin.write("END\n"); p.stdin.flush(); p.wait(timeout=2)
    except Exception:
        p.kill()


POS = "7,7,1\n5,7,2\n5,6,1\n7,6,2"

run("BOARD\n15,15,5\n" + POS + "\nDONE", "带尺寸行")
run("BOARD\n" + POS + "\nDONE", "不带尺寸行")
run("BOARD\nDONE", "空 BOARD")
