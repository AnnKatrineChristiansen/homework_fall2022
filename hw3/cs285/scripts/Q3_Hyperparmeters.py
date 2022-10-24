import shlex, subprocess, cryptography
commands = ["python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q3_0.0005 -lr 0.0005",
            "python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q3_0.005 -lr 0.005",
            "python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q3_0.01 -lr 0.01",
            "python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q3_0.05 -lr 0.05",
            "python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q3_0.1 -lr 0.1"]

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)