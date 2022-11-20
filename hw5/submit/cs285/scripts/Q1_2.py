import shlex, subprocess, cryptography
commands = ["python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --use_boltzmann --exp_name q1_alg_med",
            "python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --unsupervised_exploration --use_boltzmann --exp_name q1_alg_hard"]

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)