import os
import glob
import matplotlib.pyplot as plt

import os
import glob
import matplotlib.pyplot as plt

def save_training_curves(episode_rewards, episode_steps, losses, base_folder="training_result"):
    os.makedirs(base_folder, exist_ok=True)

    # 找出現有 trainN 資料夾編號
    existing_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f)) and f.startswith("train")]
    numbers = []
    for folder in existing_folders:
        num_str = folder.replace("train", "")
        if num_str.isdigit():
            numbers.append(int(num_str))
    next_number = max(numbers) + 1 if numbers else 1

    # 新資料夾名稱
    train_folder = os.path.join(base_folder, f"train{next_number}")
    os.makedirs(train_folder, exist_ok=True)

    # 1. 存 reward_curve.png
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(episode_rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1,2,2)
    plt.plot(episode_steps)
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")

    plt.tight_layout()
    plt.savefig(os.path.join(train_folder, "reward_curve.png"))
    plt.close()

    # 2. 存 average_reward_per_step.png
    avg_rewards = [r/s if s > 0 else 0 for r, s in zip(episode_rewards, episode_steps)]
    plt.plot(avg_rewards)
    plt.title("Average Reward per Step")
    plt.xlabel("Episode")
    plt.ylabel("Reward per Step")
    plt.savefig(os.path.join(train_folder, "average_reward_per_step.png"))
    plt.close()

    # 3. 存 looses.png
    plt.plot(losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('DQN Training Loss Curve')
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()

    print(f"Saved training curves in folder: {train_folder}")