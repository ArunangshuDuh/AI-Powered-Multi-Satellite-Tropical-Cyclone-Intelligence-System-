
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import config_reference as config
from models.cyclone_cnn import CycloneCNN

def load_data():
    try:
        train_data = torch.load(config.CACHE_DIR / "train.pt")
        val_data = torch.load(config.CACHE_DIR / "val.pt")
        
        train_ds = TensorDataset(train_data['images'], train_data['metadata'], train_data['class_labels'], train_data['reg_labels'])
        val_ds = TensorDataset(val_data['images'], val_data['metadata'], val_data['class_labels'], val_data['reg_labels'])
    except Exception as e:
        print("Could not load data from cache, generating dummy data for sanity check...", e)
        images = torch.randn(200, 5, 64, 64)
        metadata = torch.randn(200, 4)
        class_labels = torch.randint(0, 7, (200,))
        reg_labels = torch.randn(200, 2)
        
        train_ds = TensorDataset(images[:150], metadata[:150], class_labels[:150], reg_labels[:150])
        val_ds = TensorDataset(images[150:], metadata[150:], class_labels[150:], reg_labels[150:])
        val_data = {'storm_ids': [f"storm_{i//10}" for i in range(50)], 'reg_labels': reg_labels[150:]}
        
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, val_data

def main():
    train_loader, val_loader, val_data = load_data()
    
    model = CycloneCNN().to(config.DEVICE)
    
    # 3. Sanity check
    print("Running sanity check...")
    sanity_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sanity_ce = nn.CrossEntropyLoss()
    sanity_mse = nn.MSELoss()
    
    model.train()
    for epoch in range(10):
        total_loss = 0
        samples = 0
        for imgs, meta, cls_lbl, reg_lbl in train_loader:
            imgs, meta, cls_lbl, reg_lbl = imgs.to(config.DEVICE), meta.to(config.DEVICE), cls_lbl.to(config.DEVICE), reg_lbl.to(config.DEVICE)
            
            sanity_opt.zero_grad()
            cls_out, reg_out = model(imgs, meta)
            
            loss = sanity_ce(cls_out, cls_lbl) + sanity_mse(reg_out, reg_lbl)
            assert not torch.isnan(loss), "Sanity check failed: NaN loss"
            
            loss.backward()
            sanity_opt.step()
            
            total_loss += loss.item() * imgs.size(0)
            samples += imgs.size(0)
            
            if samples >= 100:
                break
                
        avg_loss = total_loss / samples
        print(f"Sanity Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
    print("Sanity check passed.")
    
    # 4. Full training
    model = CycloneCNN().to(config.DEVICE) # Reset model
    optimizer = torch.optim.AdamW(model.get_param_groups(config.LEARNING_RATE))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    lambda_reg = config.LAMBDA_REGRESSION
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        ce_loss_total = 0
        mse_loss_total = 0
        
        for imgs, meta, cls_lbl, reg_lbl in train_loader:
            imgs, meta, cls_lbl, reg_lbl = imgs.to(config.DEVICE), meta.to(config.DEVICE), cls_lbl.to(config.DEVICE), reg_lbl.to(config.DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                cls_out, reg_out = model(imgs, meta)
                ce_loss = ce_loss_fn(cls_out, cls_lbl)
                mse_loss = mse_loss_fn(reg_out, reg_lbl)
                loss = ce_loss + lambda_reg * mse_loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            ce_loss_total += ce_loss.item()
            mse_loss_total += mse_loss.item()
            
        scheduler.step()
        
        # Adjust lambda after epoch 1
        if epoch == 1:
            avg_ce = ce_loss_total / len(train_loader)
            avg_mse = mse_loss_total / len(train_loader)
            if avg_mse > 0:
                lambda_reg = avg_ce / avg_mse
                print(f"Adjusted lambda to: {lambda_reg:.4f}")
                
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        wind_mae = 0
        pressure_mae = 0
        class_correct = torch.zeros(config.NUM_CATEGORIES)
        class_total = torch.zeros(config.NUM_CATEGORIES)
        
        with torch.no_grad():
            for imgs, meta, cls_lbl, reg_lbl in val_loader:
                imgs, meta, cls_lbl, reg_lbl = imgs.to(config.DEVICE), meta.to(config.DEVICE), cls_lbl.to(config.DEVICE), reg_lbl.to(config.DEVICE)
                
                with torch.amp.autocast('cuda'):
                    cls_out, reg_out = model(imgs, meta)
                    ce = ce_loss_fn(cls_out, cls_lbl)
                    mse = mse_loss_fn(reg_out, reg_lbl)
                    loss = ce + lambda_reg * mse
                    
                val_loss += loss.item()
                preds = torch.argmax(cls_out, dim=1)
                
                correct += (preds == cls_lbl).sum().item()
                total += imgs.size(0)
                
                for c in range(config.NUM_CATEGORIES):
                    mask = (cls_lbl == c)
                    class_correct[c] += (preds[mask] == cls_lbl[mask]).sum().item()
                    class_total[c] += mask.sum().item()
                
                err = torch.abs(reg_out - reg_lbl)
                wind_mae += err[:, 0].sum().item()
                pressure_mae += err[:, 1].sum().item()
                
        val_loss /= len(val_loader)
        val_acc = correct / total
        wind_mae /= total
        pressure_mae /= total
        
        lr = optimizer.param_groups[-1]['lr']
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Wind MAE: {wind_mae:.2f} | Press MAE: {pressure_mae:.2f} | Lambda: {lambda_reg:.4f} | LR: {lr:.6f}")
        
        torch.save(model.state_dict(), config.CHECKPOINTS_DIR / f'cnn_epoch_{epoch+1}.pt')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.CHECKPOINTS_DIR / 'cnn_best.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break
                
    # 5. Compute persistence baseline
    print("Computing persistence baseline...")
    try:
        storm_ids = val_data['storm_ids']
        reg_labels = val_data['reg_labels'].cpu()
        
        pers_err = []
        for i in range(1, len(storm_ids)):
            if storm_ids[i] == storm_ids[i-1]:
                prev_val = reg_labels[i-1]
                curr_val = reg_labels[i]
                pers_err.append(torch.abs(curr_val - prev_val))
                
        if len(pers_err) > 0:
            pers_err = torch.stack(pers_err)
            pers_wind_mae = pers_err[:, 0].mean().item()
            pers_press_mae = pers_err[:, 1].mean().item()
        else:
            pers_wind_mae = 0.0
            pers_press_mae = 0.0
    except Exception:
        pers_wind_mae = 0.0
        pers_press_mae = 0.0
        
    print(f"Persistence Wind MAE: {pers_wind_mae:.2f}, Pressure MAE: {pers_press_mae:.2f}")
    
    # 6. Report final metrics
    print("\n--- Final Metrics ---")
    print(f"Classification Accuracy (Overall): {val_acc:.4f}")
    for c in range(config.NUM_CATEGORIES):
        if class_total[c] > 0:
            print(f"  Category {c} Accuracy: {class_correct[c]/class_total[c]:.4f}")
            
    print(f"Model Wind MAE: {wind_mae:.2f} | Persistence Wind MAE: {pers_wind_mae:.2f} | Delta (Model Improv): {pers_wind_mae - wind_mae:.2f}")
    print(f"Model Press MAE: {pressure_mae:.2f} | Persistence Press MAE: {pers_press_mae:.2f} | Delta (Model Improv): {pers_press_mae - pressure_mae:.2f}")

if __name__ == "__main__":
    main()
