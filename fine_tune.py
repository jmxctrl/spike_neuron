from train import load_weights, train_snn, save_weights 

W = load_weights("trained_weights/weights_reward267.4_20260317_160041.npy")

# Fine-tune with standard training (not curriculum)
# Random positions throughout, very low learning rate
W_finetuned, rewards = train_snn(
    num_episodes=100,
    learning_rate=0.0001,      # Very low for fine-tuning
    baseline_lr=0.025
)

save_weights(W_finetuned, reward=max(rewards))