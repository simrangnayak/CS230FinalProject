import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import os
import glob
import numpy as np

BATCH_SIZE = 64
LEARNING_RATE = 0.005
NUM_EPOCHS = 50

# old values: num_epochs = 50, learning_rate = 0.005, batch_size = 64

# batch_size = 32, learning_rate = 0.001, num_epochs = 100
# Final Test Accuracy: 74.38%
# Final Test F1 Score: 0.7364

# batch_size = 32, learning_rate = 0.005, num_epochs = 100
# Final Test Accuracy: 75.83%
# Final Test F1 Score: 0.7566

# batch_size = 32, learning_rate = 0.01, num_epochs = 100
# Final Test Accuracy: 77.29%
# Final Test F1 Score: 0.7701

# batch_size = 64, learning_rate = 0.005, num_epochs = 100
# Final Test Accuracy: 72.50%
# Final Test F1 Score: 0.7143


DATA_DIR = "latents" 

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

#Get the list of all composers based on file name 
def get_all_composers(directory):
    composers = set()
    for filepath in glob.glob(os.path.join(directory, "*_latents.pt")):
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        for part in parts:
            if part not in ['train', 'val', 'test', 'latents.pt', 'latents']:
                if part.lower() not in ['classical', 'jazz'] and len(part) > 2:
                     composers.add(part)
    return sorted(list(composers))

def load_and_split_data(composer_map):
    all_X = []
    all_y = []
    print(f"Aggregating data for {len(composer_map)} composers...")
    for composer, label_idx in composer_map.items():
      #Grab all files for certain composer (train, test, val)
        composer_files = glob.glob(os.path.join(DATA_DIR, f"*{composer}_latents.pt"))
        composer_latents = []
        for f in composer_files:
            try:
                data = torch.load(f, map_location=DEVICE)
                latents = data['latents'] if isinstance(data, dict) else data
                if len(latents.shape) == 1:
                    latents = latents.unsqueeze(0)
                composer_latents.append(latents)
            except Exception as e:
                print(f"  Skipping corrupt file {f}: {e}")
        if not composer_latents:
            print(f"!! WARNING: No data found for {composer}. It will be excluded.")
            continue
        full_latents = torch.cat(composer_latents, dim=0)
        #Takes all the files for that composer, creates a list of labels 
        labels = torch.full((full_latents.size(0),), label_idx, dtype=torch.long, device=DEVICE)
        all_X.append(full_latents)
        all_y.append(labels)

    #Merge & Stratified Split
    X_full = torch.cat(all_X, dim=0)
    y_full = torch.cat(all_y, dim=0)
    #80% Train, 20% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X_full.cpu().numpy(), y_full.cpu().numpy(), 
        test_size=0.2, stratify=y_full.cpu().numpy(), random_state=42)
    #Split Temp (20%) into Val (10%) and Test (10%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
        test_size=0.5, stratify=y_temp, random_state=42)

    return (torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE),
        torch.FloatTensor(X_val).to(DEVICE), torch.LongTensor(y_val).to(DEVICE),
        torch.FloatTensor(X_test).to(DEVICE), torch.LongTensor(y_test).to(DEVICE))

#Model 
class ComposerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ComposerClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    torch.manual_seed(42)

    composers = get_all_composers(DATA_DIR)
    composer_map = {name: i for i, name in enumerate(composers)}
    num_classes = len(composers)
    print(f"Found Composers: {composer_map}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data(composer_map)

    for composer, idx in composer_map.items():
        # print count in train, val, test
        train_count = (y_train == idx).sum().item()
        val_count = (y_val == idx).sum().item()
        test_count = (y_test == idx).sum().item()
        total_count = train_count + val_count + test_count
        print(f"  {composer}: Train={train_count}, Val={val_count}, Test={test_count}, Total={total_count}")

    print(f"Data Split -> Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    #Initialize model 
    model = ComposerClassifier(128, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_accuracies = []

    print("\nStarting Training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        indices = torch.randperm(X_train.size(0)).to(DEVICE)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        epoch_loss = 0
        batches = 0
        for i in range(0, X_train.size(0), BATCH_SIZE):
            bx = X_train_shuffled[i:i+BATCH_SIZE]
            by = y_train_shuffled[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        avg_loss = epoch_loss / batches
        train_losses.append(avg_loss)
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, preds = torch.max(outputs, 1)
            acc = (preds == y_val).sum().item() / y_val.size(0) * 100
            val_accuracies.append(acc)
        if (epoch+1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

    #Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)
        test_acc = (preds == y_test).sum().item() / y_test.size(0) * 100
        all_preds = preds.cpu().numpy()
        all_labels = y_test.cpu().numpy()

    #Calculate F1
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print(f"Final Test F1 Score: {f1:.4f}")

    torch.save({'model_state_dict': model.state_dict(),
        'composer_map': composer_map,'input_dim': 128,
        'num_classes': num_classes}, "composer_classifier_stratified.pth")

    #Plots 
    plt.figure(figsize=(16, 5))
    #Loss Plots 
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    #Accuracy Plot 
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, color='orange', label='Val Acc')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    #Confusion Matrix 
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    cm_percent = (cm * 100).astype(int)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=list(composer_map.keys()))
    disp.plot(ax=plt.gca(), xticks_rotation='vertical', cmap='Blues', values_format='d', colorbar=False)
    plt.title("Test Set Confusion Matrix (%)")
    plt.tight_layout()
    plt.show()
