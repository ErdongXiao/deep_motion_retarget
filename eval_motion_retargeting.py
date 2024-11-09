import torch
import torch.nn as nn
import pickle as pkl

class MotionRetargetingModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MotionRetargetingModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, :, :])
        # out = torch.sigmoid(out)
        # out = F.softmax(out, dim=1)  # Apply softmax activation along dimension 1
        return out

if __name__  == "__main__":
    file = open("wtw_ref_go1_pacing.pkl", "rb")
    resized_joint_pos = []
    data = pkl.load(file)
    for i in range(len(data["feet_pos"])-20):
        resized_joint_pos += [data["joint_pos"][i:i+20]]
    resized_joint_pos = torch.tensor(resized_joint_pos, device="cuda:0")
    # lstm_model = MotionRetargetingModel(input_size=12, hidden_size=64, num_layers=2, output_size=12)
    # lstm_model.load_state_dict(torch.load('motion_retargeting_model_from_go1_to_aliengo_len=20.pt'))
    lstm_model = torch.load('motion_retargeting_model_from_go1_to_aliengo_len=20.pt')
    print(lstm_model(resized_joint_pos))
