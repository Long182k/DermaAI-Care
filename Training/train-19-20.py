### Requirement 1: Log Five Metrics in Trainer Function
### Requirement 2: Evaluate on Hold-Out Test Set
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# Reproducibility
manualSeed = 2019
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    print("worker_seed", worker_seed)

g = torch.Generator().manual_seed(manualSeed)

# Load dataset
output_train_dir = "/kaggle/input/combined-isic-2019-2020-annotated-images/Combined_Training_By_Class"
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root=output_train_dir, transform=transform)
print("load dataset done", len(dataset))

class_names = dataset.classes
print("class_names", class_names)
class_to_idx = dataset.class_to_idx
print("Danh sách class và chỉ số tương ứng:", class_to_idx)

# Train-test split
labels = [label for _, label in dataset]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(np.zeros(len(labels)), labels):
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
print(f"Train-test split done. train_dataset = {len(train_dataset)}, test_dataset= {len(test_dataset)}")

# K-fold cross-validation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skf = StratifiedKFold(n_splits=4)
train_labels = [labels[i] for i in train_idx]
print("K-fold cross-validation done", len(train_labels), "labels")

# Build model architecture
class CNN_VGG8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.conv5 = nn.Conv2d(512, 512, 3)
        self.fc1 = nn.Linear(2048, 768)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(512, 9)
    
    def forward(self, X):
        out = F.max_pool2d(F.relu(self.conv1(X)), kernel_size=2, stride=2)
        out = F.max_pool2d(F.relu(self.conv2(out)), kernel_size=2, stride=2)
        out = F.max_pool2d(F.relu(self.conv3(out)), kernel_size=2, stride=2)
        out = F.max_pool2d(F.relu(self.conv4(out)), kernel_size=2, stride=2)
        out = F.max_pool2d(F.relu(self.conv5(out)), kernel_size=2, stride=2)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Trainer function with metrics
def trainer(model, epochs, train_data, test_data):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    best_accuracy_test = 0
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        correct = 0
        total = 0
        total_loss = 0.0
        train_label_list = []
        train_predict_list = []
        for i, data in enumerate(train_data):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            _, predicted = torch.max(out, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_label_list.extend(labels.cpu().numpy())
            train_predict_list.extend(predicted.cpu().numpy())
        
        accuracy_train = 100 * correct / total
        train_report = classification_report(train_label_list, train_predict_list, output_dict=True)
        train_precision = train_report['weighted avg']['precision']
        train_recall = train_report['weighted avg']['recall']
        train_f1 = train_report['weighted avg']['f1-score']
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_label_list = []
        val_predict_list = []
        val_probs_list = []
        with torch.no_grad():
            for i, data in enumerate(test_data):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                probs = F.softmax(out, dim=1)
                _, predicted = torch.max(out, 1)
                val_label_list.extend(labels.cpu().numpy())
                val_predict_list.extend(predicted.cpu().numpy())
                val_probs_list.extend(probs.cpu().numpy())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy_test = 100 * correct / total
        val_report = classification_report(val_label_list, val_predict_list, output_dict=True)
        val_precision = val_report['weighted avg']['precision']
        val_recall = val_report['weighted avg']['recall']
        val_f1 = val_report['weighted avg']['f1-score']
        
        # Log metrics for the epoch
        print(f"Epoch {epoch}: "
              f"Loss = {total_loss / len(train_data):.4f}, "
              f"Train Accuracy = {accuracy_train:.2f}%, "
              f"Train Precision = {train_precision:.2f}, "
              f"Train Recall = {train_recall:.2f}, "
              f"Train F1 = {train_f1:.2f}, "
              f"Test Accuracy = {accuracy_test:.2f}%, "
              f"Test Precision = {val_precision:.2f}, "
              f"Test Recall = {val_recall:.2f}, "
              f"Test F1 = {val_f1:.2f}")
        
        if accuracy_test > best_accuracy_test:
            best_accuracy_test = accuracy_test
    
    # Final validation metrics
    final_report = classification_report(val_label_list, val_predict_list, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(val_label_list, val_predict_list)
    
    # ROC Curve for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(np.array(val_label_list) == i, np.array(val_probs_list)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    return accuracy_train, best_accuracy_test, total_loss / len(train_data), final_report, conf_matrix, fpr, tpr, roc_auc

# Main loop
model_list = [CNN_VGG8() for _ in range(4)]
fold_metrics = defaultdict(list)
best_test = 0
best_fold = 0

for i_fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(np.zeros(len(train_labels)), train_labels)):
    train_indices = [train_idx[i] for i in train_fold_idx]
    val_indices = [train_idx[i] for i in val_fold_idx]
    
    fold_train_dataset = Subset(dataset, train_indices)
    fold_val_dataset = Subset(dataset, val_indices)
    
    fold_train_dataloader = DataLoader(fold_train_dataset, batch_size=16, shuffle=True, worker_init_fn=seed_worker, generator=g)
    fold_val_dataloader = DataLoader(fold_val_dataset, batch_size=16, shuffle=False, worker_init_fn=seed_worker, generator=g)
    
    print(f"\nFold {i_fold + 1}: Training...")
    
    # Train the model
    accuracy_train, accuracy_test, loss, report, conf_matrix, fpr, tpr, roc_auc = trainer(
        model=model_list[i_fold].to(device),
        epochs=50,
        train_data=fold_train_dataloader,
        test_data=fold_val_dataloader
    )
    
    # Store metrics
    fold_metrics['accuracy_train'].append(accuracy_train)
    fold_metrics['accuracy_test'].append(accuracy_test)
    fold_metrics['precision'].append(report['weighted avg']['precision'])
    fold_metrics['recall'].append(report['weighted avg']['recall'])
    fold_metrics['f1_score'].append(report['weighted avg']['f1-score'])
    fold_metrics['confusion_matrix'].append(conf_matrix)
    fold_metrics['roc_auc'].append(roc_auc)
    
    # Print fold metrics
    print(f"\nFold {i_fold + 1} Metrics:")
    print(f"Train Accuracy: {accuracy_train:.2f}%")
    print(f"Test Accuracy: {accuracy_test:.2f}%")
    print(f"Precision: {report['weighted avg']['precision']:.2f}")
    print(f"Recall: {report['weighted avg']['recall']:.2f}")
    print(f"F1 Score: {report['weighted avg']['f1-score']:.2f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_report(val_label_list, val_predict_list, target_names=class_names))
    
    # Plot ROC Curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Fold {i_fold + 1}')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_fold_{i_fold + 1}.png')
    plt.close()
    
    if accuracy_test > best_test:
        best_test = accuracy_test
        best_fold = i_fold
    
    torch.save(model_list[i_fold].state_dict(), f'/kaggle/working/cancer-model{i_fold}.pth')

# Final summary of folds
print("\nFinal Summary of All Folds:")
for i in range(4):
    print(f"\nFold {i + 1}:")
    print(f"Train Accuracy: {fold_metrics['accuracy_train'][i]:.2f}%")
    print(f"Test Accuracy: {fold_metrics['accuracy_test'][i]:.2f}%")
    print(f"Precision: {fold_metrics['precision'][i]:.2f}")
    print(f"Recall: {fold_metrics['recall'][i]:.2f}")
    print(f"F1 Score: {fold_metrics['f1_score'][i]:.2f}")

print(f"\nThe best model is from fold {best_fold + 1} with test accuracy {best_test:.2f}%")

# Evaluate on hold-out test set
print("\nEvaluating on Hold-Out Test Set...")
best_model = CNN_VGG8().to(device)
best_model.load_state_dict(torch.load(f"/kaggle/working/cancer-model{best_fold}.pth", weights_only=True))
best_model.eval()

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, worker_init_fn=seed_worker, generator=g)
correct = 0
total = 0
test_label_list = []
test_predict_list = []
test_probs_list = []

with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        out = best_model(inputs)
        probs = F.softmax(out, dim=1)
        _, predicted = torch.max(out, 1)
        test_label_list.extend(labels.cpu().numpy())
        test_predict_list.extend(predicted.cpu().numpy())
        test_probs_list.extend(probs.cpu().numpy())
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Compute test set metrics
test_accuracy = 100 * correct / total
test_report = classification_report(test_label_list, test_predict_list, target_names=class_names, output_dict=True)
test_conf_matrix = confusion_matrix(test_label_list, test_predict_list)

# ROC Curve for test set
test_fpr = {}
test_tpr = {}
test_roc_auc = {}
for i in range(len(class_names)):
    test_fpr[i], test_tpr[i], _ = roc_curve(np.array(test_label_list) == i, np.array(test_probs_list)[:, i])
    test_roc_auc[i] = auc(test_fpr[i], test_tpr[i])

# Print test set metrics
print(f"\nTest Set Metrics:")
print(f"Accuracy: {test_accuracy:.2f}%")
print(f"Precision: {test_report['weighted avg']['precision']:.2f}")
print(f"Recall: {test_report['weighted avg']['recall']:.2f}")
print(f"F1 Score: {test_report['weighted avg']['f1-score']:.2f}")
print("Confusion Matrix:\n", test_conf_matrix)
print("Classification Report:\n", classification_report(test_label_list, test_predict_list, target_names=class_names))

# Plot ROC Curve for test set
plt.figure(figsize=(10, 8))
for i in range(len(class_names)):
    plt.plot(test_fpr[i], test_tpr[i], label=f'{class_names[i]} (AUC = {test_roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Hold-Out Test Set')
plt.legend(loc="lower right")
plt.savefig('roc_curve_test_set.png')
plt.close()