import shlex, subprocess, cryptography

commands = []
iql_expectile = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
env_list = ["PointmassEasy-v0", "PointmassMedium-v0"]

for iql in iql_expectile:
    command = f"python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_supervised_lam10_tau{iql} --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --iql_expectile={iql}"
    commands.append(command)
    command = f"python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_unsupervised_lam1_tau{iql} --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile={iql}"
    commands.append(command)

    command = f"python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam50_tau{iql} --use_rnd --num_exploration_steps=20000 \--awac_lambda=50 --iql_expectile={iql}"
    commands.append(command)
    command = f"python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam1_tau{iql} --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile={iql}"
    commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)