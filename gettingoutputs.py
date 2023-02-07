import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import MyData
import numpy as np

df = pd.read_csv('dataset/newdataset.csv')
# df = df.iloc[:300000,:]
# df.to_csv('dataset/dataset4.csv', index=True)
labels = df.iloc[:,6:]
features = df.iloc[:,:6]


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        x = self.layer4(x)
        return x



def test(features,labels):
    # Load the model that we saved at the end of the training loop
    model = Network(6, 3)
    path = "trainedmodels/MLP1.pth"
    model.load_state_dict(torch.load(path))
    model.eval()


    testfeature = torch.tensor(features.values, dtype=torch.float32)
    actual_ouput = torch.tensor(labels.values, dtype=torch.float32)
    predicted_output = model(testfeature)
    

    np_input = testfeature.cpu().detach().numpy()
    np_output = predicted_output.cpu().detach().numpy()
    
    pd.DataFrame(np_output).to_csv('dataset/ypred.csv',index=False) 
    x = np.arange(0, len(actual_ouput)*0.005, 0.005)
    

    plt.figure()
    plt.subplot(311)
    plt.plot(x,np_output[:,0])
    plt.ylabel('predicted torque')
    plt.subplot(312)
    plt.plot(x,actual_ouput[:,0])
    plt.ylabel('actual torque')
    plt.subplot(313)
    plt.plot(x,testfeature[:,0])
    plt.ylabel('theta')
    plt.show()
            
def plotting(feature):
    f = torch.tensor(feature.values, dtype=torch.float32)
    f = f.cpu().detach().numpy()
    plt.figure()
    plt.plot(f, linestyle='None', marker = ".")
    plt.show()
    
    
if __name__ == "__main__":
    test(features=features, labels=labels)
    # plotting(features.iloc[:,0])
    # plotting(features.iloc[:,3])
    # plotting(features.iloc[:,0])
    print("finished")




