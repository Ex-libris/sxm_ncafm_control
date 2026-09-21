"""
measure_driver_throughput.py

Standalone, READ-ONLY diagnostic that measures the actual achievable
throughput of:
  1. The IOCTL driver (device_driver.SXMIOCTL.read_raw) - what ScopeTab's
     CaptureThread uses.
  2. The DDE channel-read path (dde_client.RealDDEClient.read_channel) -
     what the Parameters/Suggested/QPlus-Calibration tabs use for reads.

Run this ON THE PC WITH THE HARDWARE ATTACHED. It never writes/sets any
channel, parameter, or feedback state - it only reads - so it is safe to
run at any time without affecting a running experiment.

Setup:
  1. This file lives in the sxm_ncafm_control project folder, next to
     app.py, device_driver.py, dde_client.py (NOT inside the gui/ folder).
     Copy the whole project (or at least this file plus device_driver.py,
     dde_client.py, common.py, SXMRemote.py) onto the hardware PC.
  2. From that folder, run:
        python measure_driver_throughput.py
  3. Paste the console output back for interpretation, or read the
     explanation at the bottom of this file for what the numbers mean.

Expected runtime: well under a minute, unless the driver is unusually slow.
"""

import time
import statistics

import device_driver
import dde_client


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure_raw_rate(read_fn, n_samples):
    """
    Minimal-overhead timing: only a start and end timestamp, exactly like
    CaptureThread does. This gives the true achievable rate - the number
    that would actually show up in the Scope tab's status line.
    """
    t0 = time.perf_counter()
    for _ in range(n_samples):
        read_fn()
    elapsed = time.perf_counter() - t0
    rate = n_samples / elapsed if elapsed > 0 else float("inf")
    return elapsed, rate


def measure_jitter(read_fn, n_samples):
    """
    Per-call latency profile. NOTE: timing every single call adds its own
    overhead, so the aggregate rate from this pass will read LOWER than
    measure_raw_rate's number above - that's expected. This pass is only
    for looking at the *distribution* (mean/median/min/max/stdev) of
    individual call latency, i.e. how much timing jitter exists between
    consecutive samples.
    """
    latencies = []
    for _ in range(n_samples):
        t_start = time.perf_counter()
        read_fn()
        latencies.append(time.perf_counter() - t_start)
    return latencies


def report_rate(name, elapsed, rate):
    print(f"\n--- {name} ---")
    print(f"  total time:     {elapsed:.4f} s")
    print(f"  effective rate: {rate:,.1f} reads/s")


def report_jitter(name, latencies):
    n = len(latencies)
    lat_us = [l * 1e6 for l in latencies]
    print(f"\n--- {name} (jitter profile, n={n}) ---")
    print(f"  mean latency:   {statistics.mean(lat_us):.2f} us")
    print(f"  median latency: {statistics.median(lat_us):.2f} us")
    print(f"  min latency:    {min(lat_us):.2f} us")
    print(f"  max latency:    {max(lat_us):.2f} us")
    if n > 1:
        print(f"  stdev:          {statistics.stdev(lat_us):.2f} us")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SXM driver throughput diagnostic (READ-ONLY - no writes/sets)")
    print("=" * 70)

    # ---- IOCTL driver ----
    driver = None
    try:
        driver = device_driver.SXMIOCTL()
        print("\nIOCTL driver: connected OK")
    except Exception as e:
        print(f"\nIOCTL driver: FAILED to open ({e})")
        print("Skipping IOCTL measurements.")

    if driver is not None:
        chan1_name = "QPlusAmpl"
        chan2_name = "Drive"
        idx1 = device_driver.CHANNELS[chan1_name][0]
        idx2 = device_driver.CHANNELS[chan2_name][0]

        # Warm-up - first calls can be slower (paging, driver/cache warm-up)
        for _ in range(200):
            driver.read_raw(idx1)

        print(f"\nUsing channels: {chan1_name} (idx {idx1}), {chan2_name} (idx {idx2})")

        # Raw achievable rate, single channel, at increasing sample counts
        for n in (1_000, 10_000, 100_000):
            elapsed, rate = measure_raw_rate(lambda: driver.read_raw(idx1), n)
            report_rate(f"IOCTL single-channel ({chan1_name}), n={n}", elapsed, rate)

        # Raw achievable rate, two channels per iteration - matches
        # CaptureThread's actual loop structure in ScopeTab
        for n in (1_000, 10_000, 100_000):
            def two_channel_read():
                driver.read_raw(idx1)
                driver.read_raw(idx2)
            elapsed, rate = measure_raw_rate(two_channel_read, n)
            # rate here is "iterations/s"; multiply by 2 for total samples/s
            report_rate(
                f"IOCTL two-channel ({chan1_name}+{chan2_name}, matches Scope tab), n={n}",
                elapsed, rate * 2,
            )

        # Jitter profile - how uniform is the spacing between samples?
        jitter = measure_jitter(lambda: driver.read_raw(idx1), 5_000)
        report_jitter(f"IOCTL single-channel ({chan1_name})", jitter)

        driver.close()

    # ---- DDE ----
    dde = None
    try:
        dde = dde_client.RealDDEClient()
        print("\nDDE client: connected OK")
    except Exception as e:
        print(f"\nDDE client: FAILED to connect ({e})")
        print("Skipping DDE measurements.")

    if dde is not None:
        for n in (20, 100, 500):
            elapsed, rate = measure_raw_rate(lambda: dde.read_channel(0), n)
            report_rate(f"DDE read_channel(0), n={n}", elapsed, rate)

    print("\n" + "=" * 70)
    print("Done. This was READ-ONLY - no parameters were changed.")
    print("=" * 70)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# How to read the results
# ---------------------------------------------------------------------------
#
# 1. "IOCTL two-channel" rate is the number that matters for the Scope tab -
#    it's the same loop structure CaptureThread uses. Compare it to what the
#    Scope tab's status line reports during a real capture; they should be
#    close. If the real capture reports something noticeably lower, Qt/GUI
#    overhead (signal emission, plotting) is costing you more than expected.
#
# 2. Compare the n=1,000 vs n=100,000 rows for the same test. If the rate
#    stays roughly flat, the driver reaches steady throughput quickly. If it
#    drops noticeably at higher n, that's worth knowing - it could point to
#    a real degradation under sustained load, separate from anything in the
#    plotting code.
#
# 3. The two-channel rate should be close to half the single-channel rate
#    (twice the calls per sample pair). If it's much worse than half, there's
#    extra overhead/contention beyond simple call count - worth flagging.
#
# 4. Jitter profile: if stdev is a large fraction of the mean, individual
#    sample spacing is quite uneven - meaning the Scope tab's x-axis (which
#    assumes uniform spacing from the *average* rate) is only an
#    approximation, not a true fixed sample clock like a real oscilloscope.
#
# 5. DDE rate vs IOCTL rate: expect DDE to be dramatically slower (likely by
#    orders of magnitude) - it round-trips a full command string through a
#    DDE conversation with a busy-wait poll loop. This just confirms
#    ScopeTab is right to use IOCTL for capture, not DDE.
