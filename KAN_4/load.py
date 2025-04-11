import autograd.numpy as np
import torch

def load_data():
    print("Loading data from 'data_with_fields.npy' ...")
    data = np.load("KAN_4/data_with_fields.npy", allow_pickle=True).item()

    train_input = np.array(data['train_input'])   
    train_label = np.array(data['train_label'])   
    test_input  = np.array(data['test_input'])    
    test_label  = np.array(data['test_label'])    

    print(f"Number of training samples: {len(train_input)}")
    print(f"Number of test samples: {len(test_input)}\n")

    X_train = torch.tensor(train_input, dtype=torch.float32)
    y_train = torch.tensor(train_label, dtype=torch.float32)
    X_test  = torch.tensor(test_input,  dtype=torch.float32)
    y_test  = torch.tensor(test_label,  dtype=torch.float32)
    return X_train, y_train, X_test, y_test