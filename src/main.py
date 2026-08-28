import os
import subprocess
import sys

SRC  = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
PYTHON = sys.executable

FAMIGLIE = ['BA', 'ER']


def run(script, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    label = f' ({env_extra})' if env_extra else ''
    print(f'\n=== {script}{label} ===', flush=True)
    subprocess.run([PYTHON, os.path.join(SRC, script)], check=True, env=env, cwd=ROOT)


def main():
    # 1. Generazione dei dataset
    run('graphGenerator.py', {'GRAPH_TYPES': ','.join(FAMIGLIE)})

    # 2. Addestramento di Unrolling e Recurrent Unrolling per ogni famiglia
    for fam in FAMIGLIE:
        for shared in ('0', '1'):
            run('train.py', {'GRAPH_TYPE': fam, 'SHARED': shared})

    # 3. Grafici e tabelle di confronto fra famiglie
    run('curves.py')


if __name__ == '__main__':
    main()
