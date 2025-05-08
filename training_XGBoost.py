import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import joblib
import os
import shutil
from torch.amp import GradScaler, autocast
from peft import LoraConfig, get_peft_model
import xgboost as xgb
from joblib import parallel_config
import gc

# Memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

g = torch.Generator().manual_seed(manualSeed)

# Output directory
output_dir = '/home/drake/Documents/DermaAI/DermaAI-Source/Training/thesis_models_9'
os.makedirs(output_dir, exist_ok=True)

# Enhanced transform with more augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(20),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])

# Load dataset
train_dir = "/home/drake/Documents/DermaAI/DermaAI-Source/Training/data/mapping_annotated_isic_2019/Training_By_Class"
dataset = datasets.ImageFolder(root=train_dir, transform=transform)
print("Dataset loaded:", len(dataset))

class_names = dataset.classes
class_to_idx = dataset.class_to_idx
print("Class names:", class_names)
print("Class to idx:", class_to_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# K-fold cross-validation
skf = StratifiedKFold(n_splits=5)
labels = [label for _, label in dataset]
print("K-fold cross-validation prepared:", len(labels), "labels")

# Class weights
class_counts = Counter(labels)
total_samples = len(labels)
class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
weights = [class_weights[cls] for cls in sorted(class_weights.keys())]
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

def create_model():
    model = timm.create_model('seresnext101_32x4d', pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 8)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    target_modules = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d) and ('layer3' in name or 'layer4' in name)
    ]
    if not target_modules:
        raise ValueError("No Conv2d modules found in layer3 or layer4.")
    config = LoraConfig(
        target_modules=target_modules,
        r=16,  # Increased rank
        lora_alpha=32,
        lora_dropout=0.2,
    )
    model = get_peft_model(model, config)
    return model

def plot_confusion_matrix(cm, class_names, title, filename):
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def plot_roc_curve(fpr, tpr, roc_auc, class_names, title, filename):
    try:
        plt.figure(figsize=(10, 8))
        for i in range(len(class_names)):
            plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")

def extract_features(model, dataloader, batch_size=64):
    model.eval()
    features = []
    labels = []
    feature_dataloader = DataLoader(
        dataloader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=False
    )
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(feature_dataloader):
            if batch_idx % 50 == 0:
                print(f"Extracting features for batch {batch_idx}/{len(feature_dataloader)}")
            inputs = inputs.to(device)
            with autocast('cuda'):
                if isinstance(model, nn.DataParallel):
                    feats = model.module.forward_features(inputs)
                    pooled_feats = model.module.global_pool(feats)
                else:
                    feats = model.forward_features(inputs)
                    pooled_feats = model.global_pool(feats)
            features.append(pooled_feats.cpu().numpy())
            labels.append(targets.numpy())
            del inputs, feats, pooled_feats
            torch.cuda.empty_cache()
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def trainer(model, optimizer, criterion, scheduler, epochs, train_data, val_data, fold_idx):
    scaler = GradScaler()
    best_accuracy_val = 0
    patience = 5
    no_improvement_count = 0
    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        total = 0
        total_loss = 0.0
        train_label_list = []
        train_predict_list = []
        train_probs_list = []
        for i, data in enumerate(train_data):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                out = model(inputs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            probs = F.softmax(out, dim=1)
            _, predicted = torch.max(out, 1)
            train_label_list.extend(labels.cpu().numpy())
            train_predict_list.extend(predicted.cpu().numpy())
            train_probs_list.extend(probs.detach().cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
        
        accuracy_train = 100 * correct / total
        train_report = classification_report(train_label_list, train_predict_list, output_dict=True, zero_division=0)
        train_cm = confusion_matrix(train_label_list, train_predict_list)
        train_fpr = {}
        train_tpr = {}
        train_roc_auc = {}
        for i in range(len(class_names)):
            train_fpr[i], train_tpr[i], _ = roc_curve(np.array(train_label_list) == i, np.array(train_probs_list)[:, i])
            train_roc_auc[i] = auc(train_fpr[i], train_tpr[i])
        plot_confusion_matrix(train_cm, class_names,
                              f'Training Confusion Matrix - Fold {fold_idx + 1}, Epoch {epoch}',
                              f'confusion_matrix_fold_{fold_idx + 1}_epoch_{epoch}_train.png')
        plot_roc_curve(train_fpr, train_tpr, train_roc_auc, class_names,
                       f'ROC Curve (Train) - Fold {fold_idx + 1}, Epoch {epoch}',
                       f'roc_curve_fold_{fold_idx + 1}_epoch_{epoch}_train.png')
        
        model.eval()
        correct = 0
        total = 0
        val_label_list = []
        val_predict_list = []
        val_probs_list = []
        with torch.no_grad():
            for i, data in enumerate(val_data):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast('cuda'):
                    out = model(inputs)
                probs = F.softmax(out, dim=1)
                _, predicted = torch.max(out, 1)
                val_label_list.extend(labels.cpu().numpy())
                val_predict_list.extend(predicted.cpu().numpy())
                val_probs_list.extend(probs.detach().cpu().numpy())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy_val = 100 * correct / total
        val_report = classification_report(val_label_list, val_predict_list, output_dict=True, zero_division=0)
        val_cm = confusion_matrix(val_label_list, val_predict_list)
        val_fpr = {}
        val_tpr = {}
        val_roc_auc = {}
        for i in range(len(class_names)):
            val_fpr[i], val_tpr[i], _ = roc_curve(np.array(val_label_list) == i, np.array(val_probs_list)[:, i])
            val_roc_auc[i] = auc(val_fpr[i], val_tpr[i])
        plot_confusion_matrix(val_cm, class_names,
                              f'Validation Confusion Matrix - Fold {fold_idx + 1}, Epoch {epoch}',
                              f'confusion_matrix_fold_{fold_idx + 1}_epoch_{epoch}_val.png')
        plot_roc_curve(val_fpr, val_tpr, val_roc_auc, class_names,
                       f'ROC Curve (Validation) - Fold {fold_idx + 1}, Epoch {epoch}',
                       f'roc_curve_fold_{fold_idx + 1}_epoch_{epoch}_val.png')
        
        print(f"Epoch {epoch}:")
        print(f"  Training: Loss = {total_loss / len(train_data):.4f}, Accuracy = {accuracy_train:.2f}%, "
              f"Precision = {train_report['weighted avg']['precision']:.4f}, "
              f"Recall = {train_report['weighted avg']['recall']:.4f}, "
              f"F1 = {train_report['weighted avg']['f1-score']:.4f}")
        print(f"  Validation: Accuracy = {accuracy_val:.2f}%, "
              f"Precision = {val_report['weighted avg']['precision']:.4f}, "
              f"Recall = {val_report['weighted avg']['recall']:.4f}, "
              f"F1 = {val_report['weighted avg']['f1-score']:.4f}")
        
        scheduler.step(accuracy_val)
        
        if accuracy_val > best_accuracy_val:
            best_accuracy_val = accuracy_val
            no_improvement_count = 0
            try:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(output_dir, f'best_model_fold_{fold_idx}.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(output_dir, f'best_model_fold_{fold_idx}.pth'))
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        torch.cuda.empty_cache()
    
    try:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(torch.load(os.path.join(output_dir, f'best_model_fold_{fold_idx}.pth'), weights_only=True))
        else:
            model.load_state_dict(torch.load(os.path.join(output_dir, f'best_model_fold_{fold_idx}.pth'), weights_only=True))
    except Exception as e:
        print(f"Error loading model: {e}")
    model.eval()
    
    print("Starting feature extraction for training set")
    train_features, train_labels = extract_features(model, train_data, batch_size=64)
    print("Feature extraction for training set completed")
    print("Starting feature extraction for validation set")
    val_features, val_labels = extract_features(model, val_data, batch_size=64)
    print("Feature extraction for validation set completed")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    
    sample_weights = np.array([class_weights[label] for label in train_labels])
    
    print("Starting grid search for XGBoost...")
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200]
    }
    xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=8, eval_metric='mlogloss')
    grid_search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy', n_jobs=4, verbose=3)
    with parallel_config(backend='loky'):
        grid_search.fit(train_features_scaled, train_labels, sample_weight=sample_weights)
    best_clf_params = grid_search.best_params_
    print("Grid search completed. Best parameters:", best_clf_params)
    del grid_search
    gc.collect()
    
    max_depth = best_clf_params['max_depth']
    learning_rate = best_clf_params['learning_rate']
    n_estimators = best_clf_params['n_estimators']
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=8,
        eval_metric='mlogloss',
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )
    print("Starting XGBoost training")
    xgb_clf.fit(train_features_scaled, train_labels, sample_weight=sample_weights)
    print("XGBoost training completed")
    try:
        joblib.dump(scaler, os.path.join(output_dir, f'scaler_fold_{fold_idx}.pkl'))
        joblib.dump(xgb_clf, os.path.join(output_dir, f'clf_fold_{fold_idx}.pkl'))
    except Exception as e:
        print(f"Error saving scaler or classifier: {e}")
    
    print("Starting validation prediction")
    val_predict_clf = xgb_clf.predict(val_features_scaled)
    val_probs_clf = xgb_clf.predict_proba(val_features_scaled)
    print("Validation prediction completed")
    accuracy_val_clf = accuracy_score(val_labels, val_predict_clf) * 100
    val_report_clf = classification_report(val_labels, val_predict_clf, output_dict=True, zero_division=0)
    val_cm_clf = confusion_matrix(val_labels, val_predict_clf)
    val_fpr_clf = {}
    val_tpr_clf = {}
    val_roc_auc_clf = {}
    for i in range(len(class_names)):
        val_fpr_clf[i], val_tpr_clf[i], _ = roc_curve(val_labels == i, val_probs_clf[:, i])
        val_roc_auc_clf[i] = auc(val_fpr_clf[i], val_tpr_clf[i])
    plot_confusion_matrix(val_cm_clf, class_names,
                          f'Validation Confusion Matrix (XGBoost) - Fold {fold_idx + 1}',
                          f'confusion_matrix_fold_{fold_idx + 1}_clf_val.png')
    plot_roc_curve(val_fpr_clf, val_tpr_clf, val_roc_auc_clf, class_names,
                   f'ROC Curve (Validation XGBoost) - Fold {fold_idx + 1}',
                   f'roc_curve_fold_{fold_idx + 1}_clf_val.png')
    
    print(f"Fold {fold_idx + 1} XGBoost Validation Metrics: Accuracy = {accuracy_val_clf:.2f}%")
    
    return (accuracy_train, best_accuracy_val, total_loss / len(train_data), val_report, val_cm,
            val_label_list, val_predict_list, val_fpr, val_tpr, val_roc_auc,
            accuracy_val_clf, val_report_clf, val_cm_clf, val_fpr_clf, val_tpr_clf, val_roc_auc_clf)

fold_metrics = defaultdict(list)
best_val_clf = 0
best_fold = 0

for i_fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"Starting fold {i_fold + 1}")
    torch.cuda.empty_cache()
    model = create_model().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    fold_train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g, pin_memory=False
    )
    fold_val_dataloader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g, pin_memory=False
    )
    
    print(f"\nFold {i_fold + 1}: Training...")
    metrics = trainer(
        model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler, epochs=100,
        train_data=fold_train_dataloader, val_data=fold_val_dataloader, fold_idx=i_fold
    )
    
    (accuracy_train, best_accuracy_val, loss, report, conf_matrix, val_label_list, val_predict_list,
     val_fpr, val_tpr, val_roc_auc, accuracy_val_clf, val_report_clf, val_cm_clf, val_fpr_clf,
     val_tpr_clf, val_roc_auc_clf) = metrics
    
    fold_metrics['accuracy_train'].append(accuracy_train)
    fold_metrics['accuracy_val'].append(best_accuracy_val)
    fold_metrics['precision'].append(report['weighted avg']['precision'] * 100)
    fold_metrics['recall'].append(report['weighted avg']['recall'] * 100)
    fold_metrics['f1_score'].append(report['weighted avg']['f1-score'] * 100)
    fold_metrics['confusion_matrix'].append(conf_matrix)
    fold_metrics['roc_auc'].append(val_roc_auc)
    fold_metrics['accuracy_val_clf'].append(accuracy_val_clf)
    fold_metrics['precision_clf'].append(val_report_clf['weighted avg']['precision'] * 100)
    fold_metrics['recall_clf'].append(val_report_clf['weighted avg']['recall'] * 100)
    fold_metrics['f1_score_clf'].append(val_report_clf['weighted avg']['f1-score'] * 100)
    fold_metrics['confusion_matrix_clf'].append(val_cm_clf)
    fold_metrics['roc_auc_clf'].append(val_roc_auc_clf)
    
    print(f"\nFold {i_fold + 1} Metrics: Val Accuracy (Model): {best_accuracy_val:.2f}%, Val Accuracy (XGBoost): {accuracy_val_clf:.2f}%")
    if accuracy_val_clf > best_val_clf:
        best_val_clf = accuracy_val_clf
        best_fold = i_fold
    print(f"Completed fold {i_fold + 1}")

# Save the best model after cross-validation
if best_fold is not None:
    best_model_path = os.path.join(output_dir, f'best_model_fold_{best_fold}.pth')
    the_best_model_path = os.path.join(output_dir, 'the_best_model.pth')
    shutil.copy(best_model_path, the_best_model_path)
    print(f"Best model from fold {best_fold + 1} saved as 'the_best_model.pth'")

    # Print metrics for the best fold's validation set
    best_accuracy_val_clf = fold_metrics['accuracy_val_clf'][best_fold]
    best_precision_val_clf = fold_metrics['precision_clf'][best_fold]
    best_recall_val_clf = fold_metrics['recall_clf'][best_fold]
    best_f1_val_clf = fold_metrics['f1_score_clf'][best_fold]
    best_cm_val_clf = fold_metrics['confusion_matrix_clf'][best_fold]
    best_roc_auc_val_clf = fold_metrics['roc_auc_clf'][best_fold]

    # Calculate macro-average AUC for the best fold's validation set
    best_auc_val_clf = np.mean(list(best_roc_auc_val_clf.values()))

    print(f"\nBest Fold {best_fold + 1} Validation Metrics (XGBoost):")
    print(f"  Accuracy: {best_accuracy_val_clf:.2f}%")
    print(f"  Precision: {best_precision_val_clf:.2f}%")
    print(f"  Recall: {best_recall_val_clf:.2f}%")
    print(f"  F1 Score: {best_f1_val_clf:.2f}%")
    print(f"  AUC: {best_auc_val_clf:.4f}")
    print("  Confusion Matrix:\n", best_cm_val_clf)
    print("  ROC AUC:\n", best_roc_auc_val_clf)

# Test set evaluation with ensembling
test_dir = "/home/drake/Documents/DermaAI/DermaAI-Source/Training/data/annotated_test_isic_2019/ISIC_2019_Test_By_Class"
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_dataloader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g, pin_memory=False
)

test_probs_ensemble = np.zeros((len(test_dataset), 8))
for fold_idx in range(5):
    model = create_model().to(device)
    try:
        model.load_state_dict(torch.load(os.path.join(output_dir, f'best_model_fold_{fold_idx}.pth'), weights_only=True))
    except Exception as e:
        print(f"Error loading model for fold {fold_idx}: {e}")
        continue
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()
    
    try:
        scaler = joblib.load(os.path.join(output_dir, f'scaler_fold_{fold_idx}.pkl'))
        clf = joblib.load(os.path.join(output_dir, f'clf_fold_{fold_idx}.pkl'))
    except Exception as e:
        print(f"Error loading scaler or classifier for fold {fold_idx}: {e}")
        continue
    
    test_features, _ = extract_features(model, test_dataloader, batch_size=64)
    test_features_scaled = scaler.transform(test_features)
    test_probs_fold = clf.predict_proba(test_features_scaled)
    test_probs_ensemble += test_probs_fold

test_probs_ensemble /= 5
test_predict_ensemble = np.argmax(test_probs_ensemble, axis=1)
test_labels = [label for _, label in test_dataset]

test_accuracy_ensemble = accuracy_score(test_labels, test_predict_ensemble) * 100
test_report_ensemble = classification_report(test_labels, test_predict_ensemble, target_names=class_names, output_dict=True, zero_division=0)
test_conf_matrix_ensemble = confusion_matrix(test_labels, test_predict_ensemble)

test_fpr_ensemble = {}
test_tpr_ensemble = {}
test_roc_auc_ensemble = {}
for i in range(len(class_names)):
    test_fpr_ensemble[i], test_tpr_ensemble[i], _ = roc_curve(np.array(test_labels) == i, test_probs_ensemble[:, i])
    test_roc_auc_ensemble[i] = auc(test_fpr_ensemble[i], test_tpr_ensemble[i])

# Calculate macro-average AUC for the test set
test_auc_ensemble = np.mean(list(test_roc_auc_ensemble.values()))

plot_confusion_matrix(test_conf_matrix_ensemble, class_names, 'Final Confusion Matrix (Ensemble XGBoost) - Test Set', 'final_confusion_matrix_test_ensemble.png')
plot_roc_curve(test_fpr_ensemble, test_tpr_ensemble, test_roc_auc_ensemble, class_names, 'ROC Curve (Ensemble XGBoost) - Test Set', 'roc_curve_test_set_ensemble.png')

print(f"\nTest Set Metrics (Ensemble XGBoost):")
print(f"  Accuracy: {test_accuracy_ensemble:.2f}%")
print(f"  Precision: {test_report_ensemble['weighted avg']['precision'] * 100:.2f}%")
print(f"  Recall: {test_report_ensemble['weighted avg']['recall'] * 100:.2f}%")
print(f"  F1 Score: {test_report_ensemble['weighted avg']['f1-score'] * 100:.2f}%")
print(f"  AUC: {test_auc_ensemble:.4f}")
print("  Confusion Matrix:\n", test_conf_matrix_ensemble)
print("  ROC AUC:\n", test_roc_auc_ensemble)
