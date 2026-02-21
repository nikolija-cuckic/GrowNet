import torch
import torch.nn as nn

import config
from training.base_trainer import BaseTrainer


class GrowNetTrainer(BaseTrainer):
    # one epoch in base trainer = one stage/new weak learner
    def __init__(self, model, train_loader, test_loader, task_type: str = 'regression'):
        params = {
            'num_stages':  config.GROWNET_NUM_STAGES,
            'weak_lr':     config.GROWNET_WEAK_LR,
            'hidden_dim':  config.GROWNET_WEAK_HIDDEN_DIM,
            'shrinkage':   config.GROWNET_SHRINKAGE,
            'use_cs':      config.GROWNET_USE_CS,
            'cs_epochs':   config.GROWNET_CS_EPOCHS,
            'cs_every':    config.GROWNET_CS_EVERY,
            'batch_size':  config.BATCH_SIZE,
        }
        params['epochs'] = config.GROWNET_NUM_STAGES

        super().__init__(model, train_loader, test_loader, params, "GrowNet", task_type)

    #one boosting stage
    def train_epoch_step(self):
        device = self.device

        # adding new weak learner
        new_wl = self.model.add_weak_learner().to(device)

        # freezing all params except for new wl (training only on residuals)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in new_wl.parameters():
            p.requires_grad = True

        # Adam optimizator 
        optimizer = torch.optim.Adam(
            new_wl.parameters(),
            lr=self.config['weak_lr'],
            #weight_decay=0.001       # used in Badirli et al. 2020, i got worse results with it
        )

        self.model.train()
        running_loss = 0.0

        # boosting step – training new wl on residuals
        stage_index = len(self.model.models)  

        for x_batch, y_batch in self.train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # prediction of previous stages
            if stage_index == 1:
                # in first stage, output is 0
                y_pred_old = torch.zeros_like(y_batch)
                penultimate = None
            else:
                # using all except last (added new_wl)
                y_pred_old, penultimate = self.model.forward_and_features(
                    x_batch, upto=stage_index - 1
                )

            # calculating residuals
            if self.task_type == 'classification':
                # residual is probability 
                residuals = y_batch - torch.sigmoid(y_pred_old)
            else:
                residuals = y_batch - y_pred_old

            # input for new wl: x or [x, penultimate]
            if penultimate is not None:
                x_aug = torch.cat([x_batch, penultimate.detach()], dim=1)
            else:
                x_aug = x_batch

            pred_res, _ = new_wl(x_aug)

            # fitting residual as mse (first order optim)
            loss = nn.MSELoss()(pred_res, residuals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        stage_loss = running_loss / len(self.train_loader.dataset)


        # Corrective step (using all stages / whole network and fitting on real loss)
        if (self.config['use_cs'] and
                (stage_index % self.config['cs_every'] == 0)):
            for p in self.model.parameters():
                p.requires_grad = True

            cs_optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['weak_lr'],
                #weight_decay=0.001
            )

            for _ in range(self.config['cs_epochs']):
                cs_running_loss = 0.0
                for x_batch, y_batch in self.train_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                    y_pred = self.model(x_batch)
                    cs_loss = self.criterion(y_pred, y_batch)

                    cs_optimizer.zero_grad()
                    cs_loss.backward()
                    cs_optimizer.step()

                    cs_running_loss += cs_loss.item() * x_batch.size(0)

                stage_loss = cs_running_loss / len(self.train_loader.dataset)

        return stage_loss

    #early stopping and restoring best model params
    def restore_best_model(self):

        if self.best_model_state is None:
            return

        print(f"[GrowNet] Restoring best model architecture...")

        # finds max "models.X" in state_dict keys
        max_idx = -1
        for key in self.best_model_state.keys():
            if key.startswith("models."):
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    idx = int(parts[1])
                    max_idx = max(max_idx, idx)

        saved_num_models = max_idx + 1
        current_num_models = len(self.model.models)
        print(f"          Current models: {current_num_models}, Saved models: {saved_num_models}")

        if saved_num_models < current_num_models:
            new_list = nn.ModuleList()
            for i in range(saved_num_models):
                new_list.append(self.model.models[i])
            self.model.models = new_list
        elif saved_num_models > current_num_models:
            # adding empty WL-ove to have same number of slots
            for _ in range(saved_num_models - current_num_models):
                self.model.add_weak_learner()

        self.model.load_state_dict(self.best_model_state)
        self.model.to(self.device)
        print(f"[GrowNet] Successfully restored best model with {len(self.model.models)} stages.")
