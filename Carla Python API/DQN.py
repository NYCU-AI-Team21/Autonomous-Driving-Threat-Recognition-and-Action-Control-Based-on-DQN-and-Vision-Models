import torch
import torch.optim as optim
from DModel import DQN
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_size, action_size, epsilon, epsilon_min, epsilon_decay, target_update_freq, lr=1e-5):
        # Models
        self.model = DQN(state_size, action_size).to(device)
        #無yolo版 self.model = DQN(action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        #無yolo版 self.target_model = DQN(action_size).to(device)
        self.update_target_network()
        
        self.action_size = action_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.SmoothL1Loss()

         # Epsilon parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay


         # Target network update frequency
        self.target_update_freq = target_update_freq
        self.train_step_counter = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        #無yolo版 state = torch.FloatTensor(state_img).permute(2,0,1).unsqueeze(0).to(device)  # (H,W,3) -> (1,3,H,W)

        with torch.no_grad():
            q_values = self.model(state)
        action_index = q_values.argmax().item()
        return action_index  # 回傳整數

    
    def train_step(self, state, action, reward, next_state, done, gamma=0.99):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        action = torch.LongTensor([action]).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        done = torch.FloatTensor([done]).to(device)
        '''
        無yolo版
        轉換 state 和 next_state 圖片到 (1, 3, H, W)
        state = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(device)
        next_state = torch.FloatTensor(next_state).permute(2,0,1).unsqueeze(0).to(device)
        action = torch.LongTensor([action]).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        done = torch.FloatTensor([done]).to(device)
        '''

        q_values = self.model(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_state)
            next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = self.criterion(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.decay_epsilon()

        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)