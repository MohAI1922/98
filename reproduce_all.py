#!/usr/bin/env python3
"""
Master Reproduction Pipeline for Integral Derivative Manifolds (IDM & G-IDM)
Author: Muhammed M. El-Sayed

This script provides end-to-end automated verification and reproduction of:
1. All theoretical unit tests (Theorems 1-5 for Paper 1, Theorems 1-6 for Paper 2).
2. Continuous Euclidean benchmarks (Paper 1: Doppler chirp and chaotic Lorenz with colored noise).
3. Discrete non-Euclidean graph benchmarks (Paper 2: Sensor network, 3D sphere, folded Swiss roll).
4. Compilation of both LaTeX manuscripts into publication-ready PDFs.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

WORKSPACE = Path(__file__).parent.resolve()
sys.path.insert(0, str(WORKSPACE))

def run_command(cmd, desc):
    print(f"\n{'='*70}")
    print(f"[*] {desc}")
    print(f"    Command: {cmd}")
    print(f"{'='*70}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE) + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run(cmd, shell=True, cwd=str(WORKSPACE), capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if res.returncode == 0:
        print(f"[+] Success ({dt:.2f}s)")
        if res.stdout:
            lines = res.stdout.strip().split("\n")
            preview = lines if len(lines) <= 15 else lines[:8] + ["..."] + lines[-5:]
            print("\n".join("    " + l for l in preview))
        return True, res.stdout
    else:
        print(f"[-] FAILED with code {res.returncode} ({dt:.2f}s)")
        if res.stderr:
            print("    Error:\n" + "\n".join("    " + l for l in res.stderr.strip().split("\n")))
        elif res.stdout:
            print("    Output:\n" + "\n".join("    " + l for l in res.stdout.strip().split("\n")))
        return False, res.stderr or res.stdout

def main():
    print("="*70)
    print(" INTEGRAL DERIVATIVE MANIFOLDS (IDM & G-IDM) - MASTER REPRODUCTION")
    print(" Sole Author: Muhammed M. El-Sayed")
    print(f" Workspace: {WORKSPACE}")
    print("="*70)

    report_lines = []
    report_lines.append("INTEGRAL DERIVATIVE MANIFOLDS - MASTER REPRODUCTION AUDIT")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Author: Muhammed M. El-Sayed\n")

    # Step 1: Unit tests
    ok, out = run_command("python -m pytest tests/ -v", "Step 1: Running Formal Theorem Unit Tests")
    report_lines.append(f"1. Unit Tests (Theorems): {'PASS' if ok else 'FAIL'}")

    # Step 2: Paper 1 Continuous Benchmarks
    ok1, out1 = run_command("python benchmarks/run_benchmarks.py", "Step 2: Running Paper 1 Continuous Benchmarks")
    report_lines.append(f"2. Paper 1 Benchmarks (Euclidean & Lorenz): {'PASS' if ok1 else 'FAIL'}")

    # Step 3: Paper 2 Graph Benchmarks
    ok2, out2 = run_command("python benchmarks/run_graph_benchmarks.py", "Step 3: Running Paper 2 Graph & Manifold Benchmarks")
    report_lines.append(f"3. Paper 2 Benchmarks (Graph, Sphere & Swiss Roll): {'PASS' if ok2 else 'FAIL'}")

    # Step 4: LaTeX Compilation Paper 1
    ok_tex1, _ = run_command("pdflatex -interaction=nonstopmode --disable-installer paper1_manuscript.tex", "Step 4: Compiling Paper 1 LaTeX Manuscript")
    report_lines.append(f"4. Paper 1 Compilation (paper1_manuscript.pdf): {'PASS' if ok_tex1 else 'FAIL'}")

    # Step 5: LaTeX Compilation Paper 2
    ok_tex2, _ = run_command("pdflatex -interaction=nonstopmode --disable-installer paper2_manuscript.tex", "Step 5: Compiling Paper 2 LaTeX Manuscript")
    report_lines.append(f"5. Paper 2 Compilation (paper2_manuscript.pdf): {'PASS' if ok_tex2 else 'FAIL'}")

    # Summary
    all_ok = all([ok, ok1, ok2, ok_tex1, ok_tex2])
    report_lines.append(f"\nOverall Reproduction Status: {'100% REPRODUCED & VERIFIED' if all_ok else 'SOME STEPS FAILED'}")
    
    report_path = WORKSPACE / "reproduction_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[+] Master reproduction report written to: {report_path}")
    print(f"[+] Final Status: {'ALL SYSTEMS GO (100% PASS)' if all_ok else 'VERIFICATION FAILED'}\n")

if __name__ == "__main__":
    main()
