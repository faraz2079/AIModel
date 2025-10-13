# ~/pytorch/measure_grpc_with_metrics.py
import os, time, csv, threading, subprocess, math, glob

# ---- config ----
BENCH_CMD = ["python", os.path.expanduser("~/pytorch/ort_grpc_bench.py")]
SAMPLE_HZ = 1.0                           # samples per second
METRICS_CSV = os.path.expanduser("~/pytorch/system_metrics.csv")
# ----------------

stop_flag = False

def read_psi_avg10(kind):
    path = f"/proc/pressure/{kind}"
    try:
        with open(path) as f:
            txt = f.read()
        # lines look like: "some avg10=0.00 avg60=0.00 avg300=0.00 total=0"
        line = next((ln for ln in txt.splitlines() if ln.startswith("some ")), "")
        parts = dict(kv.split("=") for kv in line.replace("some ", "").split() if "=" in kv)
        return float(parts.get("avg10", "nan"))
    except Exception:
        return math.nan

def read_cpu_totals():
    # /proc/stat first line: cpu  user nice system idle iowait irq softirq steal guest guest_nice
    with open("/proc/stat") as f:
        fields = f.readline().split()
    vals = list(map(int, fields[1:8]))  # user..softirq
    idle = vals[3] + vals[4]            # idle + iowait
    nonidle = vals[0] + vals[1] + vals[2] + vals[5] + vals[6]  # user+nice+system+irq+softirq
    total = idle + nonidle
    return total, idle

def cpu_percent(prev):
    try:
        t2, i2 = read_cpu_totals()
        t1, i1 = prev if prev else (t2, i2)
        dt = max(t2 - t1, 1)
        di = i2 - i1
        pct = 100.0 * (1.0 - (di / dt))
        return pct, (t2, i2)
    except Exception:
        return math.nan, None

def rapl_energy_joules_sum():
    # sum all energy_uj files under powercap
    total_uj = 0
    for p in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"):
        try:
            with open(p) as f:
                total_uj += int(f.read().strip())
        except Exception:
            pass
    # Some systems expose domains as ...:0/*, ...:1/* etc; also try nested subdomains
    for p in glob.glob("/sys/class/powercap/intel-rapl:*/*/energy_uj"):
        try:
            with open(p) as f:
                total_uj += int(f.read().strip())
        except Exception:
            pass
    if total_uj == 0:
        return math.nan
    return total_uj / 1e6  # convert microjoules to joules

def sampler():
    global stop_flag
    interval = 1.0 / SAMPLE_HZ
    prev_cpu = None
    e0 = rapl_energy_joules_sum()
    t0 = time.time()

    # CSV header
    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "cpu_pct", "psi_cpu_avg10", "psi_mem_avg10", "psi_io_avg10", "energy_j_rel"])

    while not stop_flag:
        ts = time.time() - t0
        psi_cpu = read_psi_avg10("cpu")
        psi_mem = read_psi_avg10("memory")
        psi_io  = read_psi_avg10("io")
        cpu_pct, prev_cpu = cpu_percent(prev_cpu)
        ej = rapl_energy_joules_sum()

        energy_rel = ej - e0 if (not math.isnan(ej) and not math.isnan(e0)) else math.nan

        with open(METRICS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"{ts:.3f}",
                        f"{cpu_pct:.2f}" if not math.isnan(cpu_pct) else "",
                        f"{psi_cpu:.3f}" if not math.isnan(psi_cpu) else "",
                        f"{psi_mem:.3f}" if not math.isnan(psi_mem) else "",
                        f"{psi_io:.3f}"  if not math.isnan(psi_io)  else "",
                        f"{energy_rel:.6f}" if not math.isnan(energy_rel) else ""])

        time.sleep(interval)

def main():
    # Start sampler
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    # Run your working gRPC bench (this is the one that printed B=1..16)
    print("[metrics] starting bench:", " ".join(BENCH_CMD), flush=True)
    try:
        proc = subprocess.Popen(BENCH_CMD)
        proc.wait()
        rc = proc.returncode
    finally:
        # stop sampler
        global stop_flag
        stop_flag = True
        th.join(timeout=2.0)

    if rc != 0:
        print(f"[metrics] bench exited with code {rc}")
    else:
        print(f"[metrics] done. metrics -> {METRICS_CSV}")
        print(f"[metrics] latency -> ~/pytorch/ort_latency_grpc.csv")

if __name__ == "__main__":
    main()
