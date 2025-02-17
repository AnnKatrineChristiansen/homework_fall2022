## Part 1: “Unsupervised” RND and exploration performance

### Sub-part 1
```
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd

python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --unsupervised_exploration --exp_name q1_env1_random

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --exp_name q1_env2_random

```

### Sub-part 2
```
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --unsupervised_exploration --use_boltzmann --exp_name q1_alg_easy

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --use_boltzmann --exp_name q1_alg_med

python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --unsupervised_exploration --use_boltzmann --exp_name q1_alg_hard
```

## Part 2: Offline learning on exploration data

### Sub-part 1
```
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha 0

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha 0.1

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql_shift1_scale100 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha 0.1 --exploit_rew_shift 1 --exploit_rew_scale 100

```

### Sub-part 2
```

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps 5000 --offline_exploitation --cql_alpha 0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_5000

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps 15000 --offline_exploitation --cql_alpha 0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_15000

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps 5000 --offline_exploitation --cql_alpha 0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_5000

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps 15000 --offline_exploitation --cql_alpha 0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_15000
```

### Sub-part 3

```
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha 0.02 --exp_name q2_alpha_0.02

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha 0.5 --exp_name q2_alpha_0.5

```


## Part 3: “Supervised” exploration with mixed reward bonuses

```
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_medium_dqn

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_medium_cql

python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_hard_dqn

python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_hard_cql

```


## Part 4: Offline Learning with AWAC

For PointmassEasy-v0:
```
python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam0.1 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=0.1

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=0.1 --exp_name q4_awac_easy_supervised_lam0.1

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam1 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=1

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=1 --exp_name q4_awac_easy_supervised_lam1

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam2 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=2

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=2 --exp_name q4_awac_easy_supervised_lam2

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam10 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=10

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --exp_name q4_awac_easy_supervised_lam10

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam20 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=20

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=20 --exp_name q4_awac_easy_supervised_lam20

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam50 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=50

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --exp_name q4_awac_easy_supervised_lam50

```

For PointmassMedium-v0:
```

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam0.1 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=0.1

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=0.1 --exp_name q4_awac_medium_supervised_lam0.1

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam1 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=1

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=1 --exp_name q4_awac_medium_supervised_lam1

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam2 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=2

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=2 --exp_name q4_awac_medium_supervised_lam2

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam10 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=10

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --exp_name q4_awac_medium_supervised_lam10

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam20 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=20

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=20 --exp_name q4_awac_medium_supervised_lam20

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam50 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=50

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --exp_name q4_awac_medium_supervised_lam50

```


## Part 5


PointmassEasy-v0 Supervised: lambda = 10

```
python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_supervised_lam10_tau0.5 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --iql_expectile=0.5

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_supervised_lam10_tau0.6 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --iql_expectile=0.6

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_supervised_lam10_tau0.7 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --iql_expectile=0.7

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_supervised_lam10_tau0.8 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --iql_expectile=0.8

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_supervised_lam10_tau0.9 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --iql_expectile=0.9

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_supervised_lam10_tau0.95 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --iql_expectile=0.95

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_supervised_lam10_tau0.99 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --iql_expectile=0.99


```

PointmassEasy-v0 Unsupervised: lambda = 1

```
python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_unsupervised_lam1_tau0.5 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.5

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_unsupervised_lam1_tau0.6 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.6

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_unsupervised_lam1_tau0.7 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.7

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_unsupervised_lam1_tau0.8 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.8

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_unsupervised_lam1_tau0.9 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.9

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_unsupervised_lam1_tau0.95 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.95

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_unsupervised_lam1_tau0.99 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.99

```

PointmassMedium-v0 Supervised: lambda = 50 

```
python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam50_tau0.5 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --iql_expectile=0.5

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam50_tau0.6 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --iql_expectile=0.6

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam50_tau0.7 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --iql_expectile=0.7

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam50_tau0.8 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --iql_expectile=0.8

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam50_tau0.9 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --iql_expectile=0.9

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam50_tau0.95 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --iql_expectile=0.95

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam50_tau0.99 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --iql_expectile=0.99

```

PointmassMedium-v0 Unsupervised: lambda = 1

```

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam1_tau0.5 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.5

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam1_tau0.6 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.6

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam1_tau0.7 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.7

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam1_tau0.8 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.8

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam1_tau0.9 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.9

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam1_tau0.95 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.95

python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam1_tau0.99 --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=1 --iql_expectile=0.99

```