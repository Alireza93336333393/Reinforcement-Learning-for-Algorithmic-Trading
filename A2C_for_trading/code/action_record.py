import pandas as pd
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
class ActionRecorderCallback(BaseCallback):
    def __init__(self,action_path,verbose=1):
        super().__init__(verbose=verbose)
        self.actions = []
        if type(action_path) is str:
            self.action_path =action_path
        else:
            raise ValueError(f'the h is not str:{type(action_path)}')
     
    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )
    def _on_step(self) -> bool:

        action = self.locals['actions'] # Get the action from the locals dictionary.

    
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action[0] # Get action for the first environment if vectorized

        self.actions.append(action)
        df_actios = pd.DataFrame( np.array(self.actions))
        df_actios.to_csv(self.action_path)

        return True