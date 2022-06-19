echo "Generating hyperparameters combinations for REINFORCE, ActorCritic, PPO and TRPO"
python3 config_setter.py

echo "Testing REINFORCE..."
python3 tests_loader_reinforce.py
echo "Testing A2C..."
python3 tests_loader_ac.py
echo "Testing PPO..."
python3 tests_loader_ppo.py
echo "Testing TRPO..."
python3 tests_loader_trpo.py

echo "Retriving best configurations..."
python3 config_getter.py
echo "Best configs obtained:"
cat best_config.txt