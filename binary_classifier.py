import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os

PATHS = {'train': {
        'classical': 'train_classical_latents (1).pt',
        'jazz': 'train_jazz_latents (1).pt'},
    'val': {
        'classical': 'val_classical_latents (1).pt',
        'jazz': 'val_jazz_latents (1).pt'},
    'test': {
        'classical': 'test_classical_latents (1).pt',
        'jazz': 'test_jazz_latents (1).pt'}}

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Just did a NN of a single linear layer with a sigmoid activation function
class LogisticClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LogisticClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)
    def predict(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).float()

def load_and_prep_data(classical_path, jazz_path):
  
    if not os.path.exists(classical_path) or not os.path.exists(jazz_path):
        raise FileNotFoundError(f"Could not find files: {classical_path} or {jazz_path}")

    #load all jazz and classical
    c_latents = torch.load(classical_path, map_location=torch.device('cpu'))
    j_latents = torch.load(jazz_path, map_location=torch.device('cpu'))
    #ensure they are float tensors
    c_latents = c_latents.float()
    j_latents = j_latents.float()

    #Create Labels
    #Classical = 0
    c_labels = torch.zeros(c_latents.size(0), 1)
    #Jazz = 1
    j_labels = torch.ones(j_latents.size(0), 1)
    #Concatenate
    X = torch.cat((c_latents, j_latents), dim=0)
    y = torch.cat((c_labels, j_labels), dim=0)

    return TensorDataset(X, y)

def train_model():
    print(f"Using device: {DEVICE}")
    #Load the training data
    train_dataset = load_and_prep_data(PATHS['train']['classical'], PATHS['train']['jazz'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = load_and_prep_data(PATHS['val']['classical'], PATHS['val']['jazz'])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #Initialize model (128 is dim of latent space)
    model = LogisticClassifier(128).to(DEVICE)
    #BCEWithLogitsLoss is more stable than sigmoid and binary cross entropy loss
    criterion = nn.BCEWithLogitsLoss()
    #Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #Training loop
    train_losses = []
    val_accuracies = []

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        #Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                preds = model.predict(inputs)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
    return model, train_losses, val_accuracies


def evaluate_model(model):
    try:
        #Load the test data
        test_dataset = load_and_prep_data(PATHS['test']['classical'], PATHS['test']['jazz'])
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except FileNotFoundError:
        print("Test files not found, skipping final evaluation.")
        return
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            preds = model.predict(inputs)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())
    print(f"\nFinal Test Accuracy: {100 * correct / total:.2f}%")
    #Save the model
    torch.save(model.state_dict(), "jazz_classical_classifier.pth")

if __name__ == "__main__":
    if not os.path.exists(PATHS['train']['classical']):
        print("NOTE: Real data files not found. Please ensure paths in 'PATHS' config match your file system.")
    else:
        trained_model, loss_history, acc_history = train_model()
        evaluate_model(trained_model)
        #Plotting loss
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.legend()
        #Validation accuracy -- fluctuates a fair bit but all very high
        plt.subplot(1, 2, 2)
        plt.plot(acc_history, label='Val Accuracy', color='orange')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.show()
