from torchdiffeq import odeint
import torch
import os
import pandas as pd
import csv
from nnc.helpers.torch_utils.file_helpers import save_tensor_to_collection


class FixedInteractionEvaluator:
    def __init__(self,
                 exp_id,
                 log_dir,
                 n_interactions,
                 loss_fn=None,
                 ode_solver=None,
                 ode_solver_kwargs=dict(),
                 preserve_intermediate_states=False,
                 preserve_intermediate_controls=False,
                 preserve_intermediate_times=False,
                 preserve_intermediate_energies=False,
                 preserve_intermediate_losses=False,
                 preserve_params=False,
                 preserve_init_loss=False
                 ):
        """
        A fixed interval evaluator which evaluates a controoller over dynamics by allowing fixed
        interactions over uniformly distributed intervals in time. Within the interval the
        control signal is considered constant.
        :param exp_id: The name of the experiment, which is also generates a folder with the same
        name when logging is enabled
        :param log_dir: The path to the log directory if you enable the evaluator to log.
        :param n_interactions: The number of interations to allow within total time
        :param loss_fn: The loss function, which often is written as: `lambda t,x: scalar` which
        takes a time scalar and a state vector and returns a scalar loss.
        :param ode_solver: The ode solver to use, a function based on `torchdiffeq.odeint` such as:
        `lambda dyn, x0, t_linspace: x over steps` which takes the  dynamics function `lambda t,
        x: dx`, a batch of initial states `x0` and a linear space of `timestes` steps and returns a
        tensor of shape `[timesteps, batch_size, state_dimension]`, where state dimensions can be
        a tupple.
        :param ode_solver_kwargs: The kwards for the odesolver, especially if you want to
        generate within the evaluator.
        :param preserve_intermediate_states: Whether to preserve non final states withing
        evaluation.
        :param preserve_intermediate_controls: whether to preserve intermediate control signals
        withing evaluations.
        :param preserve_intermediate_times: whether to preserve intermediate time values.
        :param preserve_intermediate_energies: whether to preserve energy approximation through
        time.
        :param preserve_intermediate_losses: whether to preserve intermediate loss values by
        calling the loss function.
        :param preserve_params: whether to preserve `torch.nn.module.parameters` of the provided
        controller.s
        :param preserve_init_loss: whether to store loss at initial state.
        """
        self.loss_fn = loss_fn
        self.n_interactions = n_interactions
        self.exp_id = exp_id
        self.log_dir = log_dir
        self.preserve_intermediate_states = preserve_intermediate_states
        self.preserve_intermediate_controls = preserve_intermediate_controls
        self.preserve_intermediate_times = preserve_intermediate_times
        self.preserve_intermediate_energies = preserve_intermediate_energies
        self.preserve_intermediate_losses = preserve_intermediate_losses
        self.preserve_params = preserve_params
        self.preserve_init_loss = preserve_init_loss

        if ode_solver is None:
            if 'method' not in ode_solver_kwargs:
                ode_solver_kwargs['method'] = 'dopri5'
            self.ode_solver = lambda dyn, x0, t_linspace: odeint(dyn, x0, t_linspace,
                                                                 **ode_solver_kwargs)
        else:
            self.ode_solver = lambda dyn, x0, t_linspace: ode_solver(dyn, x0, t_linspace,
                                                                     **ode_solver_kwargs)
        self.ode_solver_kwargs = ode_solver_kwargs

    def evaluate(self, dynamics, controller, x0, T, epoch, progress_bar=None):
        """
        Evaluates the controller
        :param dynamics: The dynamics to evalute. the  dynamics function can be writeen as
        `lambda t, x: dx`
        :param controller: The controller function, which can be written as `lambda t, x: u`.
        :param x0: the initial state.
        :param T: The total time to evolve the state
        :param epoch: the epoch index, if the controller uses a parametrization after a given
        epoch. This is used as label so you may also use arbitary values.
        :param progress_bar: a progress bar object from `tqdm` to keep evaluation progress.
        :return: The evaluation result which is dependent on constructor parametrization.
        """
        result = dict()
        t0 = 0
        delta_t = T / self.n_interactions
        x_t = x0.detach()

        all_energy = []

        all_states = []
        all_controls = []
        all_times = []
        all_losses = []
        if self.preserve_init_loss and self.loss_fn is not None:
            loss_val = self.loss_fn(0, x_t)
            if loss_val is not None:
                all_losses.append(loss_val.cpu().detach())

        progress = range(self.n_interactions)
        if progress_bar is not None:
            progress = progress_bar(progress)

        for i in progress:
            t_start = t0 + i * delta_t
            if not isinstance(t_start, torch.Tensor):
                t_start = torch.tensor(t_start, device=x0.device)
            t_end = t_start + delta_t
            control_signal = controller(t_start, x_t).detach()
            if self.preserve_intermediate_states:
                all_states.append(x_t)
            if self.preserve_intermediate_controls:
                all_controls.append(control_signal)
            if self.preserve_intermediate_times:
                all_times.append(torch.tensor([t_start, t_end]))
            t_linspace = torch.linspace(t_start, t_end, 2, device=x0.device)
            all_energy.append((control_signal ** 2).sum(-1) * delta_t)
            cdyn = lambda t, x: dynamics(t, x, u=control_signal)
            x_t = self.ode_solver(cdyn, x_t, t_linspace)[-1]
            if self.loss_fn is not None:
                loss_val = self.loss_fn(t_end, x_t)
                if loss_val is not None:
                    all_losses.append(loss_val.cpu().detach())

        x_T = x_t

        all_energies = torch.stack(all_energy, 1).cumsum(-1)
        result['total_energy'] = all_energies[:, -1].cpu().detach()
        result['reached_state'] = x_T.cpu().detach()
        result['delta_t'] = delta_t
        result['epoch'] = epoch
        if self.loss_fn is not None:
            result['final_loss'] = all_losses[-1].cpu().detach()

        if self.preserve_intermediate_losses:
            result['all_losses'] = torch.stack(all_losses, 0).cpu().detach()
        if self.preserve_intermediate_states:
            all_states.append(x_T)
            result['all_states'] = torch.stack(all_states, 1).cpu().detach()
        if self.preserve_intermediate_controls:
            result['all_controls'] = torch.stack(all_controls, 1).cpu().detach()
        if self.preserve_intermediate_times:
            result['all_times'] = torch.stack(all_times, 1).cpu().detach()
        if self.preserve_intermediate_energies:
            result['all_energies'] = all_energies

        if self.preserve_params:
            result['nodec_params'] = controller.state_dict()
        return result

    def write_to_file(self, eval_result, other_metada={}):
        """
        Takes an evaluation result from `FixedInteractionEvaluator.evaluate` and stores is in a
        file "log_dir/exp_id".
        :param eval_result: The evaluation result dictionary.
        :param other_metada:
        :return: None
        """
        path = self.log_dir + os.path.sep + self.exp_id + os.path.sep
        os.makedirs(path, exist_ok=True)
        to_write = other_metada
        to_write['n_interactions'] = self.n_interactions
        to_write.update(self.ode_solver_kwargs)
        # preserving scalar fields in a dataframe
        fields = ['final_loss', 'total_energy', 'delta_t', 'epoch']
        for field in fields:
            if field in eval_result:
                datum = eval_result[field]
                if isinstance(datum, torch.Tensor):
                    datum = datum.cpu().detach().numpy()
                to_write[field] = datum

        df = pd.DataFrame(to_write)
        metadata_path = path + 'epoch_metadata.csv'
        if os.path.exists(metadata_path):
            df.to_csv(metadata_path, mode='a', header=False, index=False,
                      quoting=csv.QUOTE_NONNUMERIC)
        else:
            df.to_csv(metadata_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

        # preserving data that require tensor formats in a tensor file per epoch under a zip format

        tensor_datasets = ['all_losses', 'all_states', 'all_controls', 'all_times',
                           'all_energies', 'nodec_params', 'reached_state']
        for tensor_dataset in tensor_datasets:
            if tensor_dataset in eval_result:
                save_tensor_to_collection(path + 'epochs.zip',
                                          tensor_dataset + '/ep_' + str(eval_result['epoch']),
                                          eval_result[tensor_dataset]
                                          )
