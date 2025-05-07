import copy
import torch
from tqdm import tqdm

def train(model, llm, dataloader, optimizer, criterion, writer, device, epochs=10, patience=2, grad_clip=1.0, symbolic_constraints=None, checkpoint_path=None):
    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ethics, total_manip, n_batches = 0.0, 0.0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ethics_label = batch['ethics_label'].to(device)
            manipulation_label = batch['manipulation_label'].to(device)
            with torch.no_grad():
                llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
                hidden_states = llm_outputs.last_hidden_state
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(embeddings=hidden_states, attention_mask=attention_mask, symbolic_constraints=symbolic_constraints)
                    ethics_score = outputs['ethics_score']
                    manipulation_score = outputs['manipulation_score']
                    loss_ethics = criterion(ethics_score, ethics_label)
                    loss_manip = criterion(manipulation_score, manipulation_label)
                    loss = loss_ethics + 0.5 * loss_manip
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(embeddings=hidden_states, attention_mask=attention_mask, symbolic_constraints=symbolic_constraints)
                ethics_score = outputs['ethics_score']
                manipulation_score = outputs['manipulation_score']
                loss_ethics = criterion(ethics_score, ethics_label)
                loss_manip = criterion(manipulation_score, manipulation_label)
                loss = loss_ethics + 0.5 * loss_manip
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
            total_ethics += loss_ethics.item() * input_ids.size(0)
            total_manip += loss_manip.item() * input_ids.size(0)
            n_batches += input_ids.size(0)
            pbar.set_postfix({"loss": loss.item()})
        avg_loss = total_loss / len(dataloader.dataset)
        avg_ethics = total_ethics / len(dataloader.dataset)
        avg_manip = total_manip / len(dataloader.dataset)
        if writer is not None:
            writer.add_scalar('Loss/Total', avg_loss, epoch+1)
            writer.add_scalar('Loss/Ethics', avg_ethics, epoch+1)
            writer.add_scalar('Loss/Manipulation', avg_manip, epoch+1)
        # Early Stopping & Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path is not None:
                torch.save(best_model, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_model is not None:
        model.load_state_dict(best_model)
    return model 