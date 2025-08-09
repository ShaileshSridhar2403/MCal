import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np

# --- 1. Dataset Generation with a "Distractor" Class (4 Features) ---
def create_distractor_dataset(num_samples=2000):
    """
    Creates a dataset where CE loss fails on both accuracy and bias.
    Class 0: "Strong A" [1, 0, 1, 0, noise...] (Rare)
    Class 1: "Strong B" [0, 1, 0, 1, noise...] (Rare)
    Class 2: "Distractor C" [0, 0, 0, 0, noise...] (Common)
    """
    class_probs = torch.tensor([0.1, 0.1, 0.80])
    y = torch.multinomial(class_probs, num_samples, replacement=True)
    X = torch.zeros(num_samples, 6, dtype=torch.float32)
    
    mask_a = (y == 0)
    X[mask_a, 0], X[mask_a, 2] = 1.0, 1.0
    
    mask_b = (y == 1)
    X[mask_b, 1], X[mask_b, 3] = 1.0, 1.0
    
    X[:, 4:] = torch.rand(num_samples, 2) * 0.1
    return X, y

# --- 2. Model Definitions ---
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), 
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)

class Calibrator(nn.Module):
    def __init__(self, mode: str = "linear"):
        super().__init__()
        self.mode = mode
        if mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 3)
            )
        elif mode == "linear":
            self.net = nn.Linear(3, 3)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def forward(self, x):
        return self.net(x)

# --- 3. Helper Functions ---
def train_model(model, X, y, epochs=500):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

def ablate_data(X, p=0.5):
    X_ablated = X.clone()
    mask = (torch.rand(X.shape[0], 4) > p).float()
    X_ablated[:, :4] *= mask
    return X_ablated

def kl_divergence(p, q, epsilon=1e-10):
    p_stable = torch.clamp(p, min=epsilon)
    q_stable = torch.clamp(q, min=epsilon)
    return (p_stable * (p_stable.log() - q_stable.log())).sum()

# --- 4. Main Experiment ---
if __name__ == '__main__':
    modes = ["mlp", "linear"]
    for mode in modes:
        torch.manual_seed(42)
        np.random.seed(42)
        X_data, y_data = create_distractor_dataset()
        X_train, X_test = X_data[:1500], X_data[1500:]
        y_train, y_test = y_data[:1500], y_data[1500:]

        f = BaseModel()
        print("--- Training Base Model ---")
        train_model(f, X_train, y_train, epochs=500)
        f.eval()

        # Initialize all calibrators
        R_ce = Calibrator(mode=mode)
        R_kl = Calibrator(mode=mode) # The pure KL divergence calibrator
        lambda_values = [0.2, 0.4, 0.6, 0.8] 
        hybrid_calibrators = {lam: Calibrator(mode=mode) for lam in lambda_values}

        # Initialize all optimizers
        optimizer_ce = optim.Adam(R_ce.parameters(), lr=0.01)
        optimizer_kl = optim.Adam(R_kl.parameters(), lr=0.01)
        hybrid_optimizers = {lam: optim.Adam(cal.parameters(), lr=0.01) for lam, cal in hybrid_calibrators.items()}

        num_cal_epochs = 500
        print(f"\n--- Training Calibrators for {num_cal_epochs} epochs ---")
        
        for epoch in range(num_cal_epochs):
            X_masked_batch = ablate_data(X_train)
            with torch.no_grad():
                f_clean_logits = f(X_train)
                f_masked_logits = f(X_masked_batch)
                y_clean_freq = F.one_hot(f_clean_logits.argmax(dim=1), num_classes=3).float().mean(dim=0)
                y_masked_freq = F.one_hot(f_masked_logits.argmax(dim=1), num_classes=3).float().mean(dim=0)

            # --- Train R_ce (pure instance-level distillation) ---
            optimizer_ce.zero_grad()
            ce_calibrated_logits = R_ce(f_masked_logits)
            loss_ce = F.cross_entropy(ce_calibrated_logits, f_clean_logits.argmax(dim=1))
            loss_ce.backward()
            optimizer_ce.step()

            # --- Train R_kl (pure aggregate-level KL) ---
            optimizer_kl.zero_grad()
            kl_calibrated_logits = R_kl(f_masked_logits)
            loss_kl_pure = kl_divergence(F.softmax(kl_calibrated_logits, dim=1).mean(dim=0), y_clean_freq)
            loss_kl_pure.backward()
            optimizer_kl.step()

            # --- Train Hybrid Calibrators ---
            for lam in lambda_values:
                R_hybrid, optimizer_hybrid = hybrid_calibrators[lam], hybrid_optimizers[lam]
                optimizer_hybrid.zero_grad()
                calibrated_logits_hybrid = R_hybrid(f_masked_logits)
                
                loss_kl_hybrid = kl_divergence(F.softmax(calibrated_logits_hybrid, dim=1).mean(dim=0), y_clean_freq)
                loss_ce_hybrid = F.cross_entropy(calibrated_logits_hybrid, f_clean_logits.argmax(dim=1))
                loss_hybrid = (1 - lam) * loss_ce_hybrid + lam * loss_kl_hybrid
                loss_hybrid.backward()
                optimizer_hybrid.step()
                
        print("Calibrators trained.")

        print(f"\n--- Final Evaluation on Test Set with mode {R_ce.mode} ---")
        R_ce.eval()
        R_kl.eval()
        for cal in hybrid_calibrators.values():
            cal.eval()
        
        X_test_ablated = ablate_data(X_test, p=0.5)
        y_test_onehot = F.one_hot(y_test, num_classes=3).float()

        with torch.no_grad():
            base_model_logits = f(X_test_ablated)
            ce_calibrated_logits = R_ce(base_model_logits)
            kl_calibrated_logits = R_kl(base_model_logits)
            
            # Calculate probabilities
            probs_base = F.softmax(base_model_logits, dim=1)
            probs_ce = F.softmax(ce_calibrated_logits, dim=1)
            probs_kl = F.softmax(kl_calibrated_logits, dim=1)
            
            # Get predictions
            y_pred_base = probs_base.argmax(dim=1)
            y_pred_ce = probs_ce.argmax(dim=1)
            y_pred_kl = probs_kl.argmax(dim=1)

            # --- Calculate Metrics ---
            def get_metrics(y_pred, probs):
                acc = (y_pred == y_test).float().mean()
                f1 = f1_score(y_test.cpu(), y_pred.cpu(), average='macro', zero_division=0)
                brier = ((probs - y_test_onehot)**2).sum(dim=1).mean()
                ablated_dist = F.one_hot(y_pred, num_classes=3).float().mean(dim=0)
                bias = kl_divergence(ablated_dist, clean_dist)
                return acc, f1, brier, bias

            clean_dist = F.one_hot(f(X_test).argmax(dim=1), num_classes=3).float().mean(dim=0)
            
            acc_base, f1_base, brier_base, bias_base = get_metrics(y_pred_base, probs_base)
            acc_ce, f1_ce, brier_ce, bias_ce = get_metrics(y_pred_ce, probs_ce)
            acc_kl, f1_kl, brier_kl, bias_kl = get_metrics(y_pred_kl, probs_kl)

            # --- Print Results Table ---
            print("\n" + "="*70)
            print(f"Model".ljust(30) + "Accuracy".ljust(10) + "F1 Score".ljust(10) + "Brier Score".ljust(12) + "Bias (KL)")
            print("-"*70)
            print(f"Base Model".ljust(30) + f"{acc_base:.2%}".ljust(10) + f"{f1_base:.2%}".ljust(10) + f"{brier_base:.4f}".ljust(12) + f"{bias_base:.4f}")
            print(f"CE Calibrator".ljust(30) + f"{acc_ce:.2%}".ljust(10) + f"{f1_ce:.2%}".ljust(10) + f"{brier_ce:.4f}".ljust(12) + f"{bias_ce:.4f}")
            print(f"KL Calibrator".ljust(30) + f"{acc_kl:.2%}".ljust(10) + f"{f1_kl:.2%}".ljust(10) + f"{brier_kl:.4f}".ljust(12) + f"{bias_kl:.4f}")
            print("-"*70)

            # --- Hybrid Calibrator Metrics ---
            for lam in lambda_values:
                R_hybrid = hybrid_calibrators[lam]
                hybrid_calibrated_logits = R_hybrid(base_model_logits)
                probs_hybrid = F.softmax(hybrid_calibrated_logits, dim=1)
                y_pred_hybrid = probs_hybrid.argmax(dim=1)
                
                acc_hybrid, f1_hybrid, brier_hybrid, bias_hybrid = get_metrics(y_pred_hybrid, probs_hybrid)
                
                model_name = f"Hybrid (lambda={lam})"
                print(f"{model_name.ljust(30)}{acc_hybrid:.2%}".ljust(40) + f"{f1_hybrid:.2%}".ljust(10) + f"{brier_hybrid:.4f}".ljust(12) + f"{bias_hybrid:.4f}")
            print("="*70)
