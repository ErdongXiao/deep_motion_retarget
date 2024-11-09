import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle as pkl
from forward_kinematics_network import ForwardKinematicsModel
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define the LSTM model for motion retargeting
class MotionRetargetingModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MotionRetargetingModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to("cuda:0")
        self.fc = nn.Linear(hidden_size, output_size).to("cuda:0")

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to("cuda:0")
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to("cuda:0")
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, :, :])
        # out = torch.sigmoid(out)
        # out = F.softmax(out, dim=1)  # Apply softmax activation along dimension 1
        return out

# Define the custom dataset class for the motion retargeting task
class MotionRetargetingDataset(Dataset):
    def __init__(self, file_path):
        resized_joint_pos = []
        resized_feet_pos = []
        with open(file_path, 'rb') as file:
            data = pkl.load(file)
            for i in range(len(data["feet_pos"])):
                # print(data["feet_pos"][i])
                data["feet_pos"][i] = data["feet_pos"][i][0] + data["feet_pos"][i][1] + data["feet_pos"][i][2] + data["feet_pos"][i][3]
            for i in range(len(data["feet_pos"])-20):
                resized_joint_pos += [data["joint_pos"][i:i+20]]
                resized_feet_pos += [data['feet_pos'][i:i+20]]
            # self.joint_sequences = data['joint_pos']
            # self.target_foot_trajectories = data['feet_pos']
            self.joint_sequences = resized_joint_pos
            self.target_foot_trajectories = resized_feet_pos
        self.joint_sequences = torch.tensor(self.joint_sequences, dtype=torch.float32, device="cuda:0")
        self.target_foot_trajectories = torch.tensor(self.target_foot_trajectories, dtype=torch.float32, device="cuda:0")

    def __len__(self):
        return len(self.joint_sequences)

    def __getitem__(self, idx):
        return self.joint_sequences[idx], self.target_foot_trajectories[idx]

if __name__  == "__main__":
    # Define a dummy Forward Kinematics model
    model = ForwardKinematicsModel().to("cuda:0")
    model.load_state_dict(torch.load('aliengo_forward_kinematics.pt'))
    model.eval()

    # Load the dataset
    dataset = MotionRetargetingDataset('wtw_ref_go1_pacing.pkl')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Define the retargeting model, loss function, and optimizer
    lstm_model = MotionRetargetingModel(input_size=12, hidden_size=64, num_layers=1, output_size=12)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=5e-4)

    # Training the LSTM model for motion retargeting
    num_epochs = 10000
    for epoch in range(num_epochs):
        for joint_seq, target_foot_traj in dataloader:
            optimizer.zero_grad()
            # print(joint_seq.size())
            output_seq = lstm_model(joint_seq)
            
            # Calculate forward kinematics for the target robot
            forward_kinematics_model = ForwardKinematicsModel()
            predicted_foot_positions = forward_kinematics_model(output_seq)

            loss = criterion(predicted_foot_positions, (0.213 / 0.25) * target_foot_traj) + criterion(joint_seq, output_seq)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Save the trained LSTM model
    torch.save(lstm_model, 'motion_retargeting_model_from_go1_to_aliengo_len=20.pt')
