import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import random
import gc
import os
import os.path
import re
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files


def data_prepare(file_dir):
    # prepare reads from cumulus
    cumulus_seq = []
    cumulus_methy = []
    files = file_name(file_dir)
    for file in files:
        if 'cumulus' in file:  # keyword for cumulus samples for training
            print('cumulus')
            input_norm = open(file_dir + file, 'r')
            for item in input_norm:
                item = item.split('\t')
                cpg = 0
                if len(item[3]) == 66:  # check the length
                    cumulus_seq.append(item[3])
                    cumulus_methy.append(item[4])
            input_norm.close()

    # prepare reads from embryo
    embryo_seq = []
    embryo_methy = []
    files = file_name(file_dir)
    flag = 0
    for file in files:
        if 'embryo' in file: # keyword for embryo samples for training
            print('embryo')
            input_embryo = open(file_dir + file, 'r')
            for item in input_embryo:
                item = item.split('\t')
                cpg = 0
                if len(item[3]) == 66:
                    embryo_seq.append(item[3])
                    embryo_methy.append(item[4])
            input_embryo.close()
        flag = flag + 1
    return cumulus_seq, cumulus_methy, embryo_seq, embryo_methy



def lstm_seq(seq, methy):
    seq_len = len(seq)
    seq_width = len(seq[0])

    encoded_seq = np.zeros((seq_len, seq_width), dtype=int)

    encoder = LabelEncoder()
    encoder.fit(list('ACGT'))


    for i, s in enumerate(seq):
        encoded_seq[i] = encoder.transform(list(s))

    lstmseq = torch.tensor(encoded_seq, dtype=torch.int64, device=device)

    methy_tensor = torch.tensor(methy, dtype=torch.int64, device=device)
    lstmseq[methy_tensor == 1] = 4

    return lstmseq



# transform sequence into one-hot code (0/1/2/3 to one-hot) and add methylation state channel
def conv_onehot(seq):
    onehot_size = (seq.size(0), 66, 5) 
    onehot = torch.zeros(onehot_size, dtype=torch.int64, device=device)


    onehot.scatter_(2, seq.unsqueeze(-1), 1)


    methy_mask = seq == 4
    onehot[methy_mask] = torch.tensor([0, 1, 0, 0, 1], dtype=torch.int64, device=device)

    return onehot

def batched_vstack(big_tensor1, big_tensor2, batch_size):

    assert big_tensor1.size(0) == big_tensor2.size(0)
    
    stacked_tensor_list = []
    for start_idx in range(0, big_tensor1.size(0), batch_size):
        end_idx = min(start_idx + batch_size, big_tensor1.size(0))

        batch_stacked = torch.vstack((big_tensor1[start_idx:end_idx], big_tensor2[start_idx:end_idx]))
        stacked_tensor_list.append(batch_stacked)

        gc.collect()

    final_stacked_tensor = torch.vstack(stacked_tensor_list)
    return final_stacked_tensor

# Define a custom module to handle the LSTM layer
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output

class DECENT_deep(nn.Module):
    def __init__(self):
        super(DECENT_deep, self).__init__()
        self.conv1d = nn.Conv1d(5, 100, 10, padding=4)  # Adjust padding to ensure output length matches input
        self.relu1 = nn.ReLU()
        self.maxpool1d = nn.MaxPool1d(2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        self.self_attention = nn.MultiheadAttention(embed_dim=100, num_heads=4,dropout=0.2)
        self.layer_norm = nn.LayerNorm(100)
        self.custom_lstm = CustomLSTM(100, 66)  # Pass input_size and hidden_size to CustomLSTM
        self.conv1d_2 = nn.Conv1d(132, 100, 3, padding=1)  # Adjust padding to ensure output length matches input
        self.relu2 = nn.ReLU()
        self.maxpool1d_2 = nn.MaxPool1d(2, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten = nn.Flatten(1)
        self.linear1 = nn.Linear(1600, 512)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu1(x)
        x = self.maxpool1d(x)
        x = self.dropout1(x)
        x = x.permute(2, 0, 1)  # MultiheadAttention requires (seq_len, batch, embed_dim)
        x_residual=x
        x, _ = self.self_attention(x, x, x)
        x = x + x_residual  # Add the residual connection
        x = self.layer_norm(x)
        x = self.custom_lstm(x)  # Permute input to match LSTM input shape
        x = self.conv1d_2(x.permute(1, 2, 0))  # Permute input to match Conv1d input shape
        x = self.relu2(x)
        x = self.maxpool1d_2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
#        print(np.shape(x))
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.linear2(x)
#        print(np.shape(x))
        x = self.sigmoid(x)
#        print(np.shape(x))
        return x


if __name__ == '__main__':
    train_dir = '/lustre/home/2301110060/NIPGT/data/zhenyi/header/train_file/results/' # directory to store the results and data
    file_dir = '/lustre/home/2301110060/NIPGT/data/zhenyi/header/train_file/reads/' # directory of input data

    cumulus_seq, cumulus_methy, embryo_seq, embryo_methy = data_prepare(file_dir)

    print('begin_split_cumulus_meth')
    split_cumulus_methy = [[int(x) for x in line.strip()] for line in cumulus_methy]
    split_cumulus_methy2=np.array(split_cumulus_methy).astype(float)
    print('begin_cumulus_seq_lstm')
    cumulus_seq_lstm = lstm_seq(cumulus_seq, split_cumulus_methy2)


    print('begin_split_embryo_meth')
    split_embryo_methy = [[int(x) for x in line.strip()] for line in embryo_methy]
    split_embryo_methy2=np.array(split_embryo_methy).astype(float)
    print('begin_embryo_seq_lstm')
    embryo_seq_lstm = lstm_seq(embryo_seq, split_embryo_methy2)


    # randomize before class balance
    perm_0 = random.sample(range(len(cumulus_seq)), len(cumulus_seq))
    cumulus_seq_lstm = cumulus_seq_lstm[perm_0]

    print('begin_conv_onehot')
    cumulus_seq_3one_hot = conv_onehot(cumulus_seq_lstm)
    embryo_seq_3one_hot = conv_onehot(embryo_seq_lstm)

    torch.save(cumulus_seq_lstm, 'cumulus_seq_lstm.pt')
    torch.save(embryo_seq_lstm, 'embryo_seq_lstm.pt')
    torch.save(cumulus_seq_3one_hot, 'cumulus_seq_3one_hot.pt')
    torch.save(embryo_seq_3one_hot, 'embryo_seq_3one_hot.pt')



    print('begin_lstm_stack')

    data_lstm_all = torch.vstack((cumulus_seq_lstm, embryo_seq_lstm))
    print('begin_one_hot_stack')
    data = torch.vstack((cumulus_seq_3one_hot, embryo_seq_3one_hot))
    label_all = np.array([1] * len(cumulus_seq) + [0] * len(embryo_seq))  # generating labels
    

    print('begin perm')
    # randomly mixture two types of data
    perm=torch.randperm(data_lstm_all.size(0))
    print("Size of perm tensor:", perm.size())

    del cumulus_seq, cumulus_methy, embryo_seq, embryo_methy
    del cumulus_seq_lstm, embryo_seq_lstm, cumulus_seq_3one_hot, embryo_seq_3one_hot
    gc.collect()

    print('begin mixture')
    data = data[perm]
    data_lstm_all = data_lstm_all[perm]
    label_all = label_all[perm]

    del perm
    gc.collect()


    print('begin train_data')
    train_num = int(len(data) * 0.80) # 100% as training set, 80% among them for training, 20% among them for validation
    test_num = int(len(data) * 0.20)
    train_data = data[0:train_num]
    train_label = label_all[0:train_num]
    print('begin data_all')
    data_all=data[0:(train_num + test_num)]
    print('begin test_data')
    test_data = data[train_num:(train_num + test_num)]
    test_label = label_all[train_num:(train_num + test_num)]

    data_lstm = data_lstm_all[0:(train_num + test_num)]
    label = label_all[0:(train_num + test_num)]

    torch.save(train_data, 'train_data.pt')
    torch.save(train_label, 'train_label.pt')
    torch.save(test_data, 'test_data.pt')
    torch.save(test_label, 'test_label.pt')
    torch.save(label_all, 'label_all.pt')
    torch.save(data_all, 'data_all.pt')
    torch.save(data_lstm_all, 'data_lstm_all.pt')
    print('Variables saved')
    print('begin model')
    model = DECENT_deep().to(device)


    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    Batch_size=128
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1, random_state=42)
    train_dataset = TensorDataset(train_data, torch.from_numpy(train_label))
    val_dataset = TensorDataset(val_data, torch.from_numpy(val_label))
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size)
    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=Batch_size)
    dataAll_dataset = TensorDataset(data_all)
    dataAll_loader = DataLoader(dataAll_dataset, batch_size=Batch_size)
    batch_size=Batch_size
    
    history = {'train_loss': [], 'val_loss': []}  # Initialize the history dictionary
    print('begin_TRAIN')

    for epoch in tqdm(range(30)):
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            inputs = inputs.to(torch.float32) 
            outputs = model(inputs.permute(0,2,1))
            labels = labels.to(torch.float)
            loss = criterion(outputs[:,0], labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        history['train_loss'].append(train_loss/len(train_loader))

        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.to(torch.float32)  
                outputs = model(inputs.permute(0,2,1))
                labels = labels.to(torch.float)
                loss = criterion(outputs[:,0], labels)
                val_loss += loss.item()
            history['val_loss'].append(val_loss/len(val_loader))
            print(f'Epoch {epoch+1}, Train loss: {train_loss/len(train_loader)}, Validation loss: {val_loss/len(val_loader)}')

    torch.save(model.state_dict(), 'weight.pth')

    output_vectors = np.array([])
    # After training the model
    model.eval()
    with torch.no_grad():
        for inputs in test_loader:
            inputs=inputs[0]
            inputs = inputs.to(device)
            inputs = inputs.to(torch.float32)
            outputs = model(inputs.permute(0,2,1))
            output_vectors=np.append(output_vectors, outputs[:,0].cpu().numpy())

    with open(train_dir + 'training_results.txt', 'w') as f:
        f.write(str(history))  # history should be updated in your training loop

    np.savetxt(train_dir + 'predict_result.txt', output_vectors)


    output_vectors_all = np.array([])
    # After training the model
    model.eval()
    with torch.no_grad():
        for inputs in dataAll_loader:
            inputs=inputs[0]
            inputs = inputs.to(device)
            inputs = inputs.squeeze(0)
            inputs = inputs.to(torch.float32)
            outputs = model(inputs.permute(0,2,1))
            output_vectors_all=np.append(output_vectors_all, outputs[:,0].cpu().numpy())

    np.savetxt(train_dir + 'label_all.txt', label)
    np.savetxt(train_dir + 'result_all.txt', output_vectors_all)

