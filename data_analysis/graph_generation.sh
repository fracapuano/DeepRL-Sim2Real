echo "Reinforce Analysis..."
python3 deus_reinforce.py
echo "A2C and Reinforce comparison..."
python3 deus_reinforce_a2c.py
echo "PPO and TRPO comparison..."
python3 deus_ppo_trpo.py

echo "Generating pictures..."
python3 reinforce_sumup/sumup.py
python3 a2c_reinforce_sumup/sumup.py
python3 ppo_trpo_sumup/sumup.py

mv ./a2c_reinforce_sumup/action_measure_per_episode_a2cvsreinforce.svg ./images
mv ./a2c_reinforce_sumup/return_per_episode_a2cvsreinforce.svg ./images
mv ./ppo_trpo_sumup/action_measure_per_episode_ppovstrpo.svg ./images
mv ./ppo_trpo_sumup/return_per_episode_ppovstrpo.svg ./images
mv ./reinforce_sumup/action_measure_per_episode_reinforces.svg ./images
mv ./reinforce_sumup/return_per_episode_reinforces.svg ./images
