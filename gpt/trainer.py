import torch
import torch.nn.functional as F
from tqdm import tqdm # type: ignore

class Trainer:

    def __init__(self, config, model, train_loader, val_loader, eval_num_batches):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = None
        self.scheduler = None

        # set the device
        self.device = 'cuda' if config.device == 'auto' and torch.cuda.is_available() else config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        self.iter_num = 0
        self.tokens_seen = 0
        self.global_step = 0
        self.num_epochs = config.num_epochs
        self.eval_num_batches = eval_num_batches

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    def train(self, eval_freq, start_context=None, tokenizer=None):
        train_losses, val_losses, track_tokens_seen = [], [], []
        
        for epoch in range(self.num_epochs):
            self.model.train()
            for input_batch, target_batch in tqdm(self.train_loader, desc=f'Epoch: {epoch + 1} / {self.num_epochs}'):
                input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
                # zero grad, forward pass, compute loss
                self.optimizer.zero_grad()
                loss = self.compute_loss_batch(input_batch, target_batch)
                # backward propagation
                loss.backward()
                # implement clip gradient norm??
                # update grads
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                
                # track tockens seen
                self.tokens_seen += input_batch.numel()
                # track
                self.global_step += 1
                # track loss
                epoch_loss += loss.item()

                # periodic evaluation
                if self.global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model()
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(self.tokens_seen)
                    print(f"Ep {epoch+1} (Step {self.global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # generate a sample
            if start_context and tokenizer:
                print(f"Epoch {epoch + 1} completed.")
                print(self.model.generate(start_context, temperature=1.0, top_k=None))

        # compute the total number of parameters of the model
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {num_params}")

        return train_losses, val_losses, track_tokens_seen

    def compute_loss_batch(self, input_batch, target_batch):
        """compute loss for a single batch"""
        logits = self.model(input_batch) # (batch_size, block_size, vocab_size)
        return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    @torch.no_grad()
    def compute_loss_loader(self, data_loader):
        """compute the average loss over a dataloader"""
        self.model.eval()
        total_loss = 0.0
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if self.eval_num_batches and i >= self.eval_num_batches:
                break
            input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
            loss = self.compute_loss_batch(input_batch, target_batch)
            total_loss += loss.item()
            
        return total_loss / self.eval_num_batches

    @torch.no_grad()
    def evaluate_model(self):
        """evaluated the model on both training data and validation data"""
        train_loss = self.compute_loss_loader(self.train_loader)
        val_loss = self.compute_loss_loader(self.val_loader)
        return train_loss, val_loss


    