import subprocess
import signal
import sys
import time

processes = []

try:
    processes.append(
        subprocess.Popen(
            ["uv", "run", "new_main3.py"],
            cwd="/home/page/Carc2026/robot-vision-and-planning"
        )
    )

    processes.append(
        subprocess.Popen(
            ["uv", "run", "ball_align.py"],
            cwd="/home/page/Carc2026/robot-vision-and-planning"
        )
    )

    processes.append(
        subprocess.Popen(
            ["python", "movement_server.py"],
            cwd="/home/page/Carc2026"
        )
    )

    print("Waiting for services to start...")
    time.sleep(5)

    mission = subprocess.Popen(
        ["uv", "run", "mission_controller.py"],
        cwd="/home/page/Carc2026"
    )

    mission.wait()

finally:
    print("Stopping everything...")

    for p in processes:
        if p.poll() is None:
            p.terminate()

    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()