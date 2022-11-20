import shlex, subprocess, cryptography

commands = []
awac_lambda = [0.1, 1, 2, 10, 20, 50]
env_list = ["PointmassEasy-v0", "PointmassMedium-v0"]
name = ["easy", "medium"]

for num, env in enumerate(env_list):
    for lam in awac_lambda:
        command = f"python cs285/scripts/run_hw5_awac.py --env_name {env} --exp_name q4_awac_{name[num]}_unsupervised_lam{lam} --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda={lam}"
        commands.append(command)
        command = f"python cs285/scripts/run_hw5_awac.py --env_name {env} --use_rnd --num_exploration_steps=20000 --awac_lambda={lam} --exp_name q4_awac_{name[num]}_supervised_lam{lam}"
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)