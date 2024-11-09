import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
class RobotDataset(Dataset):
    def __init__(self, file_paths):
        self.joint_positions = []
        self.foot_trajectories = []
        
        for file_path in file_paths:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                self.joint_positions.extend(data['joint_pos'])
                self.foot_trajectories.extend(data['feet_pos'])

        self.joint_positions = torch.tensor(self.joint_positions, dtype=torch.float32, device="cuda:0")
        self.foot_trajectories = torch.tensor(self.foot_trajectories, dtype=torch.float32, device="cuda:0")

    def __len__(self):
        return len(self.joint_positions)

    def __getitem__(self, idx):
        return self.joint_positions[idx], self.foot_trajectories[idx]



# Define a simple neural network model using PyTorch
class ForwardKinematicsModel(nn.Module):
    def __init__(self):
        super(ForwardKinematicsModel, self).__init__()
        self.fc1 = nn.Linear(12, 64).to("cuda:0")
        self.fc2 = nn.Linear(64, 256).to("cuda:0")
        self.fc3 = nn.Linear(256, 64).to("cuda:0")
        self.fc4 = nn.Linear(64, 4 * 3).to("cuda:0")  # Output has 4 legs with 3 coordinates each

    def forward(self, x):
        x = torch.tanh(self.fc1(x)).to("cuda:0")
        x = torch.tanh(self.fc2(x)).to("cuda:0")
        x = torch.tanh(self.fc3(x)).to("cuda:0")
        x = self.fc4(x).to("cuda:0")
        return x

if __name__  == "__main__":

    # Custom dataset class for loading data

    # Define a list of file paths containing the pickle files
    # file_paths = ['wtw_ref_go1_bounding.pkl', 'wtw_ref_go1_pacing.pkl', 'wtw_ref_go1_pronking.pkl', 'wtw_ref_go1_troting.pkl']
    file_paths = ["mpc_ref_aliengo_walk_xvel=0.6.pkl"]

    # Define dataloader
    batch_size = 64
    dataset = RobotDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model, loss function, and optimizer
    model = ForwardKinematicsModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Train the model
    num_epochs = 500
    for epoch in range(num_epochs):
        for joint_angles, foot_positions in dataloader:
            optimizer.zero_grad()
            outputs = model(joint_angles)
            loss = criterion(outputs, foot_positions.view(-1, 12))  # Reshape foot_positions for comparison
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Test the model with some input joint angles
    test_angles = torch.tensor([0., 0.9, -1.8, 0., 0.9, -1.8, 0., 0.9, -1.8, 0., 0.9, -1.8], dtype=torch.float32)
    # test_angles = torch.tensor(np.random.uniform(low=-np.pi, high=np.pi, size=(1, 12)), dtype=torch.float32)
    # predicted_positions = model(test_angles)
    # print("Predicted Foot Positions:")
    # print(predicted_positions.view(4, 3))  # Reshape to get the positions for each foot

    # Save the trained model
    # torch.save(model.state_dict(), 'go1_forward_kinematics.pt')
    torch.save(model.state_dict(), 'aliengo_forward_kinematics.pt')
    # Load the model for re-evaluation
    model = ForwardKinematicsModel()
    # model.load_state_dict(torch.load('go1_forward_kinematics.pt'))
    model.load_state_dict(torch.load('aliengo_forward_kinematics.pt'))
    model.eval()

    # Test the model with some input joint angles
    predicted_positions = model(test_angles)
    print("Predicted Foot Positions:")
    print(predicted_positions.view(4, 3))  # Reshape to get the positions for each foot