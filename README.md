# EC754-project
### Requirements
Tensorflow 1.13, Z3

### Required steps 
- cd to the root folder of this project
- export PYTHONPATH=$(pwd)/gym/gym/envs/classic_control

### Test a policy
- cd cartpole-v0
- python3 test.py --total_steps=500 --discrete --no_intervals=32

### Visualize a run of a policy trained on discrete or continuous states
- vlc discrete.mp4
- vlc continuous.mp4

### Generate a transition model (requires a lot of RAM)
python3 generate_transition_model_cartpole.py --no_intervals=10 --episodes=1

### Produce a formula by running
python3 create_formula_cartpole.py --x0=-0.3951612903225805 --x1=-0.3951612903225805 --x2=0.05000000000000028 --x3=-0.3209677419354837 --model_id=500 --timesteps=2 --mdp=mdp_32_1.npy --no_intervals=32

### Run Z3
time z3 formulas/x0_-0.3951612903225805x1_-0.3951612903225805x2_0.05000000000000028x3_-0.3209677419354837/cartpole_500_2_mdp_32_1.npy.smt2 > out2
