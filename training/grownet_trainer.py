import torch
import torch.nn as nn
import config
from training.base_trainer import BaseTrainer

class GrowNetTrainer(BaseTrainer):
    def __init__(self, model, train_loader, test_loader, task_type='regression'):
        """
        Args:
            model: GrowNet (nn.ModuleList kontejner)
            train_loader: DataLoader za trening
            test_loader: DataLoader za test
            task_type: 'regression' ili 'classification'
        """
        params = {
            'num_stages': config.GROWNET_NUM_STAGES,
            'weak_lr': config.GROWNET_WEAK_LR,
            'hidden_dim': config.GROWNET_WEAK_HIDDEN_DIM,
            'shrinkage': config.GROWNET_SHRINKAGE,
            'use_cs': config.GROWNET_USE_CS,
            'cs_epochs': config.GROWNET_CS_EPOCHS,
            'cs_every': config.GROWNET_CS_EVERY
        }
        
        # Override config['epochs'] jer BaseTrainer koristi taj ključ za petlju
        # Za GrowNet, "epochs" u BaseTraineru postaju "Stages"
        params['epochs'] = config.GROWNET_NUM_STAGES

        super().__init__(model, train_loader, test_loader, params, "GrowNet", task_type)

    def train_epoch_step(self):
        """
        Ova metoda se poziva u petlji BaseTrainer-a.
        Za GrowNet, jedan 'epoch' je zapravo jedan 'stage'.
        
        Koraci:
        1. Dodaj novi Weak Learner (WL).
        2. Treniraj samo taj WL na rezidualima (y - y_current).
        3. (Opciono) Corrective Step: Treniraj ceo model na originalnom y.
        """
        
        # 1. Dodaj novi Weak Learner u model
        # Pretpostavljamo da tvoj model ima metodu .add_weak_learner()
        # i da vraca referencu na taj novi sloj
        new_wl = self.model.add_weak_learner()
        new_wl.to(self.device)

        # Zamrzni sve prethodne, odrzni samo novi
        for p in self.model.parameters():
            p.requires_grad = False
        for p in new_wl.parameters():
            p.requires_grad = True

        # Optimizer samo za novi WL
        optimizer = torch.optim.Adam(new_wl.parameters(), lr=config.GROWNET_WEAK_LR, weight_decay=0.001)

        # -------------------------------------------------------------
        # 2. Boosting Step: Treniraj novi WL na rezidualima
        # -------------------------------------------------------------
        self.model.train()
        running_loss = 0.0

        for x_batch, y_batch in self.train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            # Izracunaj trenutnu predikciju BEZ novog WL-a
            # (GrowNet model forward obicno sabira sve, pa moramo peske ili imati metodu)
            # Najlakse: model(x) - shrinkage * novi_wl(x)
            with torch.no_grad():
                # Ovo radi ako model.forward() vec ukljucuje novi_wl koji je dodat
                current_full_pred = self.model(x_batch)
                # Oduzmi doprinos novog netreniranog WL-a da dobijes "stari" output
                # Pazi: Ako je new_wl tek inicijalizovan, on daje neki random sum.
                # Bolji pristup: model(x) vraca sumu svih. 
                # Ali new_wl je vec dodat u listu.
                
                # Formula iz rada: residuals = gradients of Loss(y, y_pred_old)
                # Za MSE: residual = y - y_pred_old
                y_pred_old = current_full_pred - config.GROWNET_SHRINKAGE * new_wl(x_batch)

            # Cilj za novi WL: residual
            # POPRAVKA ZA 1st ORDER GRADIENT BOOSTING
            if self.task_type == 'classification':
                # Rezidual = y - prob (NE y - logit)
                # y_pred_old su logiti
                residuals = y_batch - torch.sigmoid(y_pred_old)
            else:
                residuals = y_batch - y_pred_old

            
            # Forward novog WL
            pred_res = new_wl(x_batch)
            
            # Loss: Koliko dobro novi WL pogadja reziduale
            # (Koristimo MSE i za klasifikaciju u ovom koraku jer fitujemo gradient!)
            loss = nn.MSELoss()(pred_res, residuals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        # -------------------------------------------------------------
        # 3. Corrective Step (Opciono, ali preporuceno)
        # -------------------------------------------------------------
        # Treniramo CEO model na originalnom targetu (y_batch)
        if config.GROWNET_USE_CS and (len(self.model.models) % config.GROWNET_CS_EVERY == 0):
            
            # Odmrznemo sve parametre
            for p in self.model.parameters():
                p.requires_grad = True
                
            cs_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.GROWNET_WEAK_LR, weight_decay=0.001)
            
            # Vrtimo CS nekoliko epoha (definisano u configu)
            for _ in range(config.GROWNET_CS_EPOCHS):
                cs_running_loss = 0.0
                for x_batch, y_batch in self.train_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                    y_pred = self.model(x_batch)
                    
                    # Ovde koristimo PRAVI loss (MSE za reg, BCE za class)
                    # self.criterion je setovan u BaseTrainer
                    cs_loss = self.criterion(y_pred, y_batch)

                    cs_optimizer.zero_grad()
                    cs_loss.backward()
                    cs_optimizer.step()
                    
                    cs_running_loss += cs_loss.item() * x_batch.size(0)
                
                # Azuriramo running_loss da odrazava stanje posle CS-a
                running_loss = cs_running_loss

        # Vracamo finalni loss ove epohe (stage-a)
        stage_loss = running_loss / len(self.train_loader.dataset)
        return stage_loss

    def restore_best_model(self):
        """
        Specifična logika za GrowNet:
        Moramo prilagoditi broj Weak Learner-a u modelu onome što je sačuvano u best_model_state.
        """
        if self.best_model_state is None:
            return

        print(f"[GrowNet] Restoring best model architecture...")
        
        # 1. Izbroj koliko modela ima u sacuvanom state_dict-u
        # Trazimo kljuceve oblika 'models.0.net...', 'models.1.net...'
        # Najveci broj X u 'models.X' nam govori koliko ih ima.
        max_idx = -1
        for key in self.best_model_state.keys():
            if key.startswith("models."):
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    idx = int(parts[1])
                    if idx > max_idx:
                        max_idx = idx
        
        saved_num_models = max_idx + 1
        current_num_models = len(self.model.models)
        
        print(f"          Current models: {current_num_models}, Saved models: {saved_num_models}")

        # 2. Prilagodi self.model
        if saved_num_models < current_num_models:
            # Ako model ima vise slojeva nego sto treba (jer smo nastavili trening posle najboljeg),
            # brisemo visak.
            # nn.ModuleList nema .resize(), pa moramo rucno.
            new_list = nn.ModuleList()
            for i in range(saved_num_models):
                new_list.append(self.model.models[i])
            self.model.models = new_list
            
        elif saved_num_models > current_num_models:
            # Ovo se retko desava u GrowNet treningu, ali za svaki slucaj:
            # Morali bismo da dodamo prazne WL-ove.
            for _ in range(saved_num_models - current_num_models):
                self.model.add_weak_learner()
        
        # 3. Sada mozemo bezbedno ucitati tezine
        self.model.load_state_dict(self.best_model_state)
        self.model.to(self.device)
        print(f"[GrowNet] Successfully restored best model with {len(self.model.models)} stages.")
