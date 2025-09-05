import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertModel, XLMRobertaForSequenceClassification, XLMRobertaModel, Trainer, TrainerCallback

import torch.nn.functional as F
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(preds, labels[0], average='macro')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

class EarlyStoppingEpochCallback(TrainerCallback):
    def __init__(self, early_stopping_patience):
        self.early_stopping_patience = early_stopping_patience
        self.best_loss = None
        self.patience_counter = 0

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.log_history:
            current_loss = state.log_history[-1].get('eval_loss')
            if current_loss is not None:
                if self.best_loss is None or current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"No improvement in evaluation loss for {self.early_stopping_patience} epochs. Stopping training.")
                        control.should_training_stop = True