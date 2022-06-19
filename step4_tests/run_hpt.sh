echo "Generating hyperparameters combinations for TRPO"
python3 config_setter.py

echo "Testing TRPO..."
python3 tests_loader_trpo.py

echo "Retriving best configurations..."
python3 config_getter.py
echo "Best configs obtained:"
cat best_config.txt