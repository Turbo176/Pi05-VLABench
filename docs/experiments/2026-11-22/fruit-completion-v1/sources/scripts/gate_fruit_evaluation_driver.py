"""Prevent the serial driver from duplicating a concurrently scored language stage."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time


def process_alive(pid):
    path = Path('/proc') / str(pid) / 'stat'
    try:
        return path.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--driver-pid', type=int, required=True)
    parser.add_argument('--job-pid', type=int, required=True)
    parser.add_argument('--pause-during', choices=('ordinary-ppo', 'fruit_language-sft'), required=True)
    parser.add_argument('--wait-for', choices=('fruit_language-base', 'fruit_language-ppo'), required=True)
    args = parser.parse_args()
    driver = Path('/proc') / str(args.driver_pid)
    command = (driver / 'cmdline').read_bytes().split(b'\0')
    if command[:2] != [b'bash', b'code/scripts/run-fruit-completion-resume-v1.sh']:
        raise ValueError('The driver PID does not belong to this experiment')
    destination = args.root / ('driver-gate-' + args.wait_for + '.json')
    expected_output = str(args.root / args.pause_during).encode()
    report = dict(driver_pid=args.driver_pid, concurrent_job_pid=args.job_pid,
                  pause_during=args.pause_during, wait_for=args.wait_for, events=[])
    paused = False

    def record(event):
        report['events'].append(dict(at=datetime.now(timezone.utc).isoformat(), event=event))
        destination.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report['events'][-1]), flush=True)

    def terminate(signum, frame):
        raise SystemExit('Gate received termination signal')

    signal.signal(signal.SIGTERM, terminate)
    try:
        record('Waiting for the specified child evaluation; no policy process will be paused.')
        while not (args.root / args.wait_for / 'completed.json').exists():
            if not process_alive(args.driver_pid) or not process_alive(args.job_pid):
                raise RuntimeError('The serial driver or concurrent evaluation exited before completion')
            children = (driver / 'task' / str(args.driver_pid) / 'children').read_text().split()
            for child in children:
                try:
                    parts = (Path('/proc') / child / 'cmdline').read_bytes().split(b'\0')
                except FileNotFoundError:
                    continue
                if (b'code/scripts/evaluate_pose_development.py' in parts and b'--output' in parts
                        and parts[parts.index(b'--output') + 1] == expected_output):
                    os.kill(args.driver_pid, signal.SIGSTOP)
                    paused = True
                    record('Paused only the shell driver; its current model evaluation continues.')
                    break
            if paused:
                break
            time.sleep(2)
        while not (args.root / args.wait_for / 'completed.json').exists():
            if not process_alive(args.job_pid):
                raise RuntimeError('The concurrent evaluation failed; completion was not assumed')
            time.sleep(2)
        record('Concurrent evaluation completed; the serial driver can skip the verified group.')
    finally:
        if paused and process_alive(args.driver_pid):
            os.kill(args.driver_pid, signal.SIGCONT)
            record('Resumed the shell driver.')


if __name__ == '__main__':
    main()
