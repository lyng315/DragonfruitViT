import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import get_dataloaders
from models.vgg16_model import get_vgg16_model
from models.resnet50_model import get_resnet50_model
from models.mobilenetv2_model import get_mobilenetv2_model
from models.vit_model import get_vit_model

CHECKPOINTS_DIR = "checkpoints"
OUTPUTS_DIR     = "outputs"

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(OUTPUTS_DIR, f'confusion_matrix_{model_name}.png'))
    plt.close()

def train_model(model_name, num_epochs=50, learning_rate=0.001, batch_size=32):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR,     exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(batch_size=batch_size)
    num_classes = len(class_names)

    if model_name == 'vgg16':
        model = get_vgg16_model(num_classes)
    elif model_name == 'resnet50':
        model = get_resnet50_model(num_classes)
    elif model_name == 'mobilenetv2':
        model = get_mobilenetv2_model(num_classes)
    elif model_name == 'vit':
        model = get_vit_model(num_classes)
    else:
        raise ValueError("Invalid model name!")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    history = []
    best_acc = 0.0
    patience_early_stop = 7
    epochs_no_improve = 0
    
    # --- TRAINING LOOP ---
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_corrects.double().item() / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} F1: {f1:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': val_acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        })

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, f"best_{model_name}.pth"))
            save_confusion_matrix(all_labels, all_preds, class_names, model_name)
        else:
            epochs_no_improve += 1

        scheduler.step(epoch_val_loss)

        if epochs_no_improve >= patience_early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # --- FINAL EVALUATION ---
    print(f"\n{'='*20} FINAL TEST EVALUATION {'='*20}")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, f"best_{model_name}.pth")))
    model.eval()
    
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro', zero_division=0)

    print(f"Model: {model_name}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test F1-Score : {t_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))

    pd.DataFrame(history).to_csv(os.path.join(OUTPUTS_DIR, f"history_{model_name}.csv"), index=False)
    
    test_results = {
        'model': model_name,
        'test_acc': test_acc,
        'test_precision': t_prec,
        'test_recall': t_rec,
        'test_f1': t_f1
    }
    pd.DataFrame([test_results]).to_csv(os.path.join(OUTPUTS_DIR, f"test_results_{model_name}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline for Dragon Fruit Disease Classification")
    parser.add_argument('--model', type=str, default='mobilenetv2', help='vgg16, resnet50, mobilenetv2, vit')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    train_model(model_name=args.model, num_epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)
