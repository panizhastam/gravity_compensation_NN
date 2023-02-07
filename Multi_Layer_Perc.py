import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import MyData
import numpy as np
from tqdm import tqdm

df = pd.read_csv('dataset/newdataset.csv')

train, test = train_test_split(df, test_size=0.2, random_state=42)
test.to_csv('dataset/testdata.csv', index=False)
np.shape(test)

train, validation = train_test_split(train, test_size=0.2, random_state=42)
train.to_csv('dataset/traindata.csv', index=False)
validation.to_csv('dataset/validationdata.csv', index=False)
np.shape(validation)

learning_rate = 1e-6
batch_size = 25

path = 'dataset/traindata.csv'
train_data = MyData.GainDataset(path)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
print("The tensor shape in a training set is: ", len(train_loader) * batch_size)

path = 'dataset/testdata.csv'
test_data = MyData.GainDataset(path)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
print("The tensor shape in a test set is: ", len(test_loader) * batch_size)

path = 'dataset/validationdata.csv'
valid_data = MyData.GainDataset(path)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
print("The tensor shape in a valid set is: ", len(valid_loader) * batch_size)

input_size = 6
output_size = 3

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

# Instantiate the model
model = Network(input_size, output_size)


def saveModel():
    path = "trainedmodels/MLP1.pth"
    torch.save(model.state_dict(), path)
    
    
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)



def set_globvar():
    global best_val_error    # Needed to modify global copy of best_val_error
    best_val_error = 1e10


def train(num_epochs):
    best_val_error = 1e10
    print("Begin training...")
    for epoch in range(1, num_epochs + 1):
        running_train_loss = 0.0 
        running_accuracy = 0.0 
        running_vall_loss = 0.0 
        total = 0 

        # Training Loop
        for data in train_loader:
            # for data in enumerate(train_loader, 0):
            inputs, outputs = data
            optimizer.zero_grad()  # zero the parameter gradients
            predicted_outputs = model(inputs)  # predict output from the model
            train_loss = loss_fn(predicted_outputs, outputs)  # calculate loss for the predicted output
            train_loss.backward()  # backpropagation
            optimizer.step()  # adjust parameters based on the calculated gradients
            # print('epoch {}, train loss {}'.format(epoch, train_loss.data))
            running_train_loss += train_loss.item()  # track the loss value

        # Calculate training loss value
        train_loss_value = running_train_loss / len(train_loader)
        print('Completed training batch', epoch, 'Training Loss is: %.4f' % train_loss_value)
        
        # Validation Loop 
        with torch.no_grad(): 
            model.eval() 
            for data in valid_loader: 
               inputs, outputs = data 
               predicted_outputs = model(inputs) 
               val_loss = loss_fn(predicted_outputs, outputs) 
            #    print('epoch {}, validation loss {}'.format(epoch, val_loss.data))
               running_vall_loss += val_loss.item()  
               running_accuracy += running_vall_loss
 
        # Calculate validation loss value 
        val_loss_value = running_vall_loss/len(valid_loader) 
        print('Completed validation batch', epoch, 'Validation Loss is: %.4f' % val_loss_value)
        
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
        accuracy = running_accuracy/len(valid_loader)
    
        # print("Average accuracy: %f" % accuracy)
        # print("Test count error: %f" % val_loss_value/250)    
 
        # Save the model if the accuracy is the best 
        if val_loss_value < best_val_error:
            best_val_error = val_loss_value

            saveModel()

        # Print the statistics of the epoch
        
def test():
    # Load the model that we saved at the end of the training loop
    model = Network(input_size, output_size)
    path = "trainedmodels/MLP1.pth"
    model.load_state_dict(torch.load(path))
    model.eval()
    running_accuracy = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            outputs = outputs.to(torch.float32)
            predicted_outputs = model(inputs)
            error = predicted_outputs - outputs # honestly I dont know what to do with the error nor how to interpret it haha:))
            running_accuracy = mean_squared_error(outputs, predicted_outputs)

        print('Accuracy of the model based on the test set of', len(test_loader) * batch_size,
              'inputs is: %d %%' % running_accuracy) # this should somehow be in percents


        
        
if __name__ == "__main__":
    torch.manual_seed(42)
    num_epochs = 60
    train(num_epochs)
    print('Finished Training\n')
    test()