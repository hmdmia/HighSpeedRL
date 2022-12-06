python_a4h
This repository is a combined Python and Matlab repository for the Autonomy for Hypersonics (A4H) project sponsored by Sandia.
The code in this repository can:
1. Train RL agents to make an emergency descent of a hypersonic glide vehicle to a desired altitude via a closed-loop guidance law that chooses Angle of Attack based on the current state.
2. Train RL agents via Motion Primitives (MPs) to hit a non-maneuvering moving target without a-priori information of target speed and heading
3. Unsuccessfully train RL agent without MPs to hit a non-maneuvering moving target without a-priori information of target speed and heading
4. Tune hyperparameters via Bayesian optimization for the above 3 scenarios.

These goals are achieved using Python-only. An intermediate solution to 2. exists in a combined MATLAB/Python environment, which requires the MATLAB repository DAF in order to run.

Python Environment
To run the code in this repository requires the "bleeding-edge" version of Stable-baselines3 (SB3) and Optuna. SB3 requires Python>=3.7. Optuna requires Python>=3.6.

Create Virtual Environment with Conda
conda create --no-default-packages --name a4h
conda activate a4h
conda install python
conda install pip
pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
pip install optuna

Create Virtual Environment with Python>=3.7 on your path
python -m venv a4h

source a4h/bin/activate For bash
a4h/bin/Activate.ps1 PowerShell Core
a4h\Scripts\Activate.ps1 Windows PowerShell

pip install pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
pip install optuna

DAF
The DAF repository must be in the parent folder of python_a4h to be recognized by the MATLAB scripts in ./+a4h and ./+daf_sim. i.e. the relative path to DAF must be "../daf" to run the scripts.

Most Pertinent Scripts to Run
For each script, ensure your "working directory" is python_a4h. This can be done in console by running "pythonPath.sh" in bash. This can be done in Pycharm by setting the configuration working direction to python_a4h.

Train RL agent to make emergency descent of hypersonic glide vehicle:
Nominal training script:
./rl_runners/aoa_runners/ppo_aoa_absolute.py

To train with normally distrubuted starting position:
./rl_runners/aoa_runners/ppo_aoa_random_start.py

To tune new hyperparameters:
./rl_runners/aoa_runners/tune_ppo.py

Train RL agents to via MPs to hit non-maneuvering moving targets
Nominal training script:
./rl_runners/mp_runners/moving_target_training.py

To tune new hyperparameters:
./rl_runners/mp_runners/movingTargetTune.py

To evaluate a trained agent in an example run:
./rl_runners/mp_runners/run_saved_model.py

To train against a stationary target without including target movement in agent's observation:
./rl_runners/mp_runners/ppo_mp_fpa_trim_and_turn.py

To train against a stationary target with a random initial state:
./rl_runners/mp_runners/ppo_mp_fpa_trim_and_turn_random.py

To tune new stationary target hyperparameters:
./rl_runners/mp_runners/tune_ppo_mp.py

Train RL agents without MPs to hit non-maneuvering moving targets
Nominal training script:
./rl_runners/non_mp_runners/non_mp_movingtarget.py

To tune new hyperparameters:
./rl_runners/non_mp_runners/nonmp_tune.py

To evaluate a trained agent in an example run:
./rl_runners/non_mp_runners/run_saved_model.py

Train RL agents with DAF via MPs to hit non-maneuvering moving targets
Nominal training script (run in terminal):
./dafhgvmpTest.sh

Run in debug mode (extra terminal output):
./dafhgvmpTestDebug.sh

Tune new hyperparameters:
./dafhgvmpTune.sh

Run separately in an open Python and MATLAB GUI:
In MATLAB, run (50000 is port number, False is bool for debug mode):
addpath('../daf'); daf_sim.DafServer.run(50000, false);

In Python, run:
rl_runners/dafHGVmpTest.py

NOTE: MATLAB and Python communicate on a Port in a server/client relationship.
To change port number, modify "port" file (open in text editor).

Overview of Backend Code

./+a4h
The ./+agents Folder has the HGVs and target models. Interfacing with RL occurs here. However, MovementManager handles EoMs for HGV.
The ./+analysis Folder has scripts for post-processing and plotting of DAF runs
The ./+funcs Folder has the movement manager which has EoMs for HGV and implementation of MPs to get a control input to EoMs.
The ./+runners Folder has the different runners the server will run based on what the Python client requests.
The ./+util folder has support classes for atmosphere, earth, state, and vehicle type classes.

./+daf_sim
Houses the DafServer (which interacts with Python Client and runs appropriate MATLAB scripts for training, tuning, evaluation, etc.)

./backend/base_aircraft_classes
This folder has vehicle classes to implement EoMs, integration, and event handling.

./backend/call_backs
This folder has files for tensorboard callbacks for RL training

./backend/data_handling
This folder has files for loading/storing beluga trajectories

./backend/rl_base_classes
This folder has classes that extend the base_aircraft_classes to implement a reward, observation, reset, and step functions.

./backend/rl_environments
This folder has classes that can take an rl_base_class as a parameter to initialize a Box, Discrete, or MultiDiscrete environment for SB3.

./backend/utils
This folder has miscellanious support scripts such as hyperparameter saving/loading/handling, atmosphere modeling, the Python Client for DAF, and the logger for DAF.

./canonical
This folder has saved canonical models for the descent problem and non-maneuvering moving target problem used in the IEEE and AIAA papers.

The emergency descent agent is: ./canonical/9_23_21/ppo_aoa_random_start.zip
The Python/DAF agent is: ./canoncial/10_29_2021_hgvmp/dhm1to500mps800k.zip
The MP moving target Python agent is: ./canonical/05_16_22_moving_target/canoncial_0to500v_pm180psi.zip
The non-MP moving target Python agent is: ./canoncal/05_17_22_non_mp/canonical_0to500v_pm180psi.zip

./model_testing
This folder has some intermediate script used in development of the code to evaluate the base aircraft and RL classes

./optimal_references
This folder has beluga-generated json files for the emergency descent problem

./results_visualization
This folder has files to evaluate trained agents with one or multiple runs.

./saved_hyperparameters
This folder has text files with the hyperparameters tuned to each problem via Optuna.
