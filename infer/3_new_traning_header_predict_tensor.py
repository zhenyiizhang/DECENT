import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import gc
import os.path
import re
import sklearn
import time
from sklearn.preprocessing import LabelEncoder
import os


import argparse

parser = argparse.ArgumentParser(description='validation.')
parser.add_argument('--file_dir', type=str, help='Directory containing file')
parser.add_argument('--store_dir', type=str, help='Directory to store')
args = parser.parse_args()




file_dir = args.file_dir
store_dir = args.store_dir
os.makedirs(store_dir, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files




# transform sequence into number for storage (A/T/C/G to 0/1/2/3, methylated C to 4)
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


def make_chrom_region(chrom_0, region_0):
    i = 0
    chrom = np.zeros(len(chrom_0), dtype='int')
    region = np.zeros(len(region_0), dtype='int')
    while i < len(chrom_0):
        matches = re.findall('(\d+)', chrom_0[i])
        if matches:
            chrom[i] = int(matches[0])
        else:
#            print(f"No digits found in {chrom_0[i]} at index {i}")
            chrom_0[i]=-1

        region[i] = region_0[i]
        i = i + 1
    return chrom, region


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
    model = DECENT_deep().to(device)
    model.load_state_dict(torch.load('weight1kw_new.pth',map_location=torch.device('cpu')))

    files = file_name(file_dir)
    for file in files:
        input = open(file_dir + file,'r')
        seq = []
        methy = []
        headers=[]
        for item in input:
            item = item.split('\t')  
            header = item[0] 
            seq.append(item[3])
            methy.append(item[4])
            headers.append(header)  

        input.close()
        print('begin_split_meth')
        split_methy = [[int(x) for x in line.strip()] for line in methy]
        split_methy2=np.array(split_methy).astype(float)
        print('begin_lstm')
        seq_lstm = lstm_seq(seq, split_methy2)
        print('begin_onehot')
        seq_3one_hot = conv_onehot(seq_lstm)
        dataAll_dataset = TensorDataset(seq_3one_hot)
        dataAll_loader = DataLoader(dataAll_dataset, batch_size=10000)
        output_vectors_all = np.array([])
    # After training the model
        model.eval()
        print('begin_eval')
        with torch.no_grad():
            for inputs in dataAll_loader:
                inputs=inputs[0]
                inputs = inputs.to(device)
                inputs = inputs.squeeze(0)
                inputs = inputs.to(torch.float32)
                outputs = model(inputs.permute(0,2,1))
                output_vectors_all=np.append(output_vectors_all, outputs[:,0].cpu().numpy())

            
        np.savetxt(os.path.join(store_dir, 'result_' + file + '.txt'), output_vectors_all)
        with open(os.path.join(store_dir, 'header_' + file + '.txt'), 'w') as header_file:
            for header in headers:
                header_file.write(header + '\n')

#    output_vectors_all=np.array(output_vectors_all)
#    print(np.shape(output_vectors_all))
#    output_vectors_reshaped_all = output_vectors_all.reshape(-1, 1)
    # result_all = model.predict(data[0:(train_num + test_num)], verbose=0)
