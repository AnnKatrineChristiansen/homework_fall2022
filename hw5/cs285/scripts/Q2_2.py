import shlex, subprocess, cryptography
commands = ["python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps 5000 --offline_exploitation --cql_alpha 0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_5000",
            "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps 15000 --offline_exploitation --cql_alpha 0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_15000",
            "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps 5000 --offline_exploitation --cql_alpha 0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_5000",
            "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps 15000 --offline_exploitation --cql_alpha 0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_15000"]

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)