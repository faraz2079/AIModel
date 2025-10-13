#!/usr/bin/env python3
# measure_grpc_with_metrics.py
import os, time, csv, threading, subprocess, math, glob, re, shutil
from pathlib import Path

# ---- config ----
BASE_DIR    = Path(__file__).resolve().parent
BENCH_CMD   = ["python", str(BASE_DIR / "ort_grpc_bench.py")]  # run locally
SAMPLE_HZ   = 1.0                                              # samples per second
METRICS_DIR = BASE_DIR / "metrics"
# The bench writes here by default; we will move/rename this after it finishes
LATENCY_TMP = BASE_DIR / "ort_latency_grpc.csv"
# ----------------

stop_flag = False

def next_run_index():
    """Find max index among existing metrics files and return next (starting from 1)."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    idx = 0
    pat = re.compile(r".*_(\d+)\.csv$")
    for p in METRICS_DIR.glob("*.csv"):
        m = pat.match(p.name)
        if m:
            try:
                idx = max(idx, int(m.group(1)))
            except ValueError:
                pass
    return (idx + 1) if idx >= 1 else 1

def read_psi_avg10(kind):
    path = f"/proc/pressure/{kind}"
    try:
        with open(path) as f:
            txt = f.read()
        line = next((ln for ln in txt.splitlines() if ln.startswith("some ")), "")
        parts = dict(kv.split("=") for kv in line.replace("some ", "").split() if "=" in kv)
        return float(parts.get("avg10", "nan"))
    except Exception:
        return math.nan

def read_cpu_totals():
    with open("/proc/stat") as f:
        fields = f.readline().split()
    vals = list(map(int, fields[1:8]))  # user..softirq
    idle = vals[3] + vals[4]            # idle + iowait
    nonidle = vals[0] + vals[1] + vals[2] + vals[5] + vals[6]
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
    total_uj = 0
    for p in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"):
        try:
            with open(p) as f:
                total_uj += int(f.read().strip())
        except Exception:
            pass
    for p in glob.glob("/sys/class/powercap/intel-rapl:*/*/energy_uj"):
        try:
            with open(p) as f:
                total_uj += int(f.read().strip())
        except Exception:
            pass
    if total_uj == 0:
        return math.nan
    return total_uj / 1e6  # microjoules -> joules

def sampler(metrics_csv_path: Path):
    global stop_flag
    interval = 1.0 / SAMPLE_HZ
    prev_cpu = None
    e0 = rapl_energy_joules_sum()
    t0 = time.time()

    # CSV header
    with open(metrics_csv_path, "w", newline="") as f:
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

        with open(metrics_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                f"{ts:.3f}",
                f"{cpu_pct:.2f}" if not math.isnan(cpu_pct) else "",
                f"{psi_cpu:.3f}" if not math.isnan(psi_cpu) else "",
                f"{psi_mem:.3f}" if not math.isnan(psi_mem) else "",
                f"{psi_io:.3f}"  if not math.isnan(psi_io)  else "",
                f"{energy_rel:.6f}" if not math.isnan(energy_rel) else ""
            ])
        time.sleep(interval)

def main():
    run_idx = next_run_index()
    metrics_csv = METRICS_DIR / f"system_metrics_{run_idx}.csv"
    latency_csv = METRICS_DIR / f"ort_latency_grpc_{run_idx}.csv"

    # Start sampler thread
    th = threading.Thread(target=sampler, args=(metrics_csv,), daemon=True)
    th.start()

    # Run benchmark (it will create BASE_DIR/ort_latency_grpc.csv)
    print("[metrics] starting bench:", " ".join(BENCH_CMD), flush=True)
    try:
        proc = subprocess.Popen(BENCH_CMD, cwd=str(BASE_DIR))
        proc.wait()
        rc = proc.returncode
    finally:
        global stop_flag
        stop_flag = True
        th.join(timeout=2.0)

    # Move/rename latency CSV with index if it exists
    if LATENCY_TMP.exists():
        try:
            shutil.move(str(LATENCY_TMP), str(latency_csv))
        except Exception:
            # Fallback: copy & remove
            shutil.copy2(str(LATENCY_TMP), str(latency_csv))
            try: LATENCY_TMP.unlink()
            except Exception: pass

    if rc != 0:
        print(f"[metrics] bench exited with code {rc}")
    else:
        print(f"[metrics] done.")
        print(f" - metrics -> {metrics_csv}")
        print(f" - latency -> {latency_csv if latency_csv.exists() else '(not produced)'}")

if __name__ == "__main__":
    main()
