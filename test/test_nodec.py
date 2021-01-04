import unittest
import torch
from torchdiffeq import odeint
from scipy.integrate import ode
from tqdm.auto import tqdm

from nnc.helpers.torch_utils.evaluators import FixedInteractionEvaluator
from nnc.helpers.torch_utils.graphs import maximum_matching_drivers, adjacency_tensor, drivers_to_tensor
from nnc.controllers.baselines.ct_lti.dynamics import ContinuousTimeInvariantDynamics
from nnc.controllers.baselines.ct_lti.optimal_controllers import ControllabiltyGrammianController
from nnc.helpers.torch_utils.nn_architectures.fully_connected import StackedDenseTimeControl
from nnc.helpers.torch_utils.trainers import NODECTrainer
from nnc.controllers.neural_network.nnc_controllers import NNCDynamics
from nnc.helpers.torch_utils.losses import FinalStepMSE
from nnc.helpers.data_generators import cca_state_generator
import networkx as nx

class NODECTester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_nodes = 4
        self.graph = nx.random_tree(self.n_nodes, seed=1)
        self.adjacency_m = adjacency_tensor(self.graph)
        self.driver_m = drivers_to_tensor(self.n_nodes, maximum_matching_drivers(self.graph))
        self.n_drivers = self.driver_m.shape[-1]

        self.total_time = 0.5

        torch.manual_seed(1)
        self.x0 = torch.rand([1, self.n_nodes])
        self.x_target = cca_state_generator(self.x0[0], self.adjacency_m,  3)

    def test_ct_lti_dynamics(self):

        dyn = ContinuousTimeInvariantDynamics(self.adjacency_m.to(torch.float64),
                                              self.driver_m.to(torch.float64),
                                              dtype=torch.float64
                                              )

        np_dyns = lambda t, y: dyn(t, torch.tensor(y, dtype=torch.float64)).cpu().detach().numpy()

        scp_odeint = ode(np_dyns).set_integrator('dopri5')
        scp_odeint.set_initial_value(self.x0[0].cpu().numpy(), 0)
        t_space = torch.linspace(0,  self.total_time, 2).to(torch.float64)

        x_tdq = odeint(dyn, self.x0.to(torch.float64), t_space, method='dopri5')
        x_scp = scp_odeint.integrate(self.total_time)
        self.assertTrue(torch.allclose(x_tdq[-1, 0, :], torch.tensor(x_scp)))

    def test_ct_lti_oc(self):
        dyn = ContinuousTimeInvariantDynamics(self.adjacency_m.to(torch.float64),
                                              self.driver_m.to(torch.float64))
        oc = ControllabiltyGrammianController(self.adjacency_m.to(torch.float64),
                                              self.driver_m.to(torch.float64),
                                              self.total_time,
                                              self.x0.to(torch.float64),
                                              self.x_target.to(torch.float64),
                                              100
                                              )
        np_dyns = lambda t, y: dyn(t,
                                   torch.tensor(y, dtype=torch.float64),
                                   u=oc(torch.tensor([t])[0].to(torch.float64),
                                        torch.tensor(y, dtype=torch.float64)
                                        )
                                   ).cpu().detach().numpy()

        scp_odeint = ode(np_dyns).set_integrator('vode')
        scp_odeint.set_initial_value(self.x0[0].cpu().numpy(), 0)
        x_scp = scp_odeint.integrate(self.total_time)
        x_target = self.x_target[0].to(torch.float64)
        rmae = torch.abs((x_target - torch.tensor(x_scp)) / x_target).mean()
        self.assertLessEqual(rmae, 10**-3)

    def test_ct_lti_nodec(self):
        dyn = ContinuousTimeInvariantDynamics(self.adjacency_m.to(torch.float64),
                                              self.driver_m.to(torch.float64))
        torch.manual_seed(1)
        nn = StackedDenseTimeControl(self.n_nodes, self.n_drivers, n_hidden=1,
                            hidden_size=self.n_nodes+4,
                                     activation=torch.nn.functional.elu)\
            .to(torch.float64)
        nndyn = NNCDynamics(dyn, nn)
        trainer = NODECTrainer(nndyn,
                     self.x0.to(torch.float64),
                     self.x_target.to(torch.float64),
                     self.total_time,
                     optimizer_class=torch.optim.LBFGS,
                     optimizer_params= {'lr': 0.1,
                        'max_iter' : 1,
                        'max_eval' : 1,
                        'history_size':100
                       },
                      ode_solver_kwargs=dict(method='dopri5'),
                      use_adjoint=False
                     )
        nndyn = trainer.train_best(epochs=500,
                                   lr_adapation_rate=0.5,
                                   loss_variance_tolerance=10
                                   )

        np_dyns = lambda t, y: dyn(t,
                                   torch.tensor(y, dtype=torch.float64),
                                   u=nndyn.nnc(torch.tensor([t])[0].to(torch.float64),
                                               torch.tensor(y, dtype=torch.float64)
                                               )
                                   ).cpu().detach().numpy()

        scp_odeint = ode(np_dyns).set_integrator('vode')
        scp_odeint.set_initial_value(self.x0[0].cpu().numpy(), 0)
        x_scp = scp_odeint.integrate(self.total_time)
        x_target = self.x_target[0].to(torch.float64)
        rmae = torch.abs((x_target - torch.tensor(x_scp))/x_target).mean()
        self.assertLessEqual(rmae, 10**-3)

    def test_trainer_logs(self):
        dyn = ContinuousTimeInvariantDynamics(self.adjacency_m.to(torch.float64),
                                              self.driver_m.to(torch.float64))
        torch.manual_seed(1)
        nn = StackedDenseTimeControl(self.n_nodes, self.n_drivers, n_hidden=1,
                                     hidden_size=self.n_nodes + 4,
                                     activation=torch.nn.functional.elu) \
            .to(torch.float64)
        nndyn = NNCDynamics(dyn, nn)
        trainer = NODECTrainer(nndyn,
                               self.x0.to(torch.float64),
                               self.x_target.to(torch.float64),
                               self.total_time,
                               optimizer_class=torch.optim.LBFGS,
                               optimizer_params={'lr': 0.1,
                                                 'max_iter': 1,
                                                 'max_eval': 1,
                                                 'history_size': 100
                                                 },
                               ode_solver_kwargs=dict(method='dopri5'),
                               use_adjoint=False
                               )
        nndyn = trainer.train_best(epochs=500,
                                   lr_adapation_rate=0.5,
                                   loss_variance_tolerance=10
                                   )

        np_dyns = lambda t, y: dyn(t,
                                   torch.tensor(y, dtype=torch.float64),
                                   u=nndyn.nnc(torch.tensor([t])[0].to(torch.float64),
                                               torch.tensor(y, dtype=torch.float64)
                                               )
                                   ).cpu().detach().numpy()

    def test_evaluator(self):
        dyn = ContinuousTimeInvariantDynamics(self.adjacency_m,
                                              self.driver_m)
        torch.manual_seed(1)
        nn = StackedDenseTimeControl(self.n_nodes, self.n_drivers, n_hidden=1,
                                     hidden_size=self.n_nodes + 4,
                                     activation=torch.nn.functional.elu) \

        nndyn = NNCDynamics(dyn, nn)
        trainer = NODECTrainer(nndyn,
                               self.x0,
                               self.x_target,
                               self.total_time,
                               optimizer_class=torch.optim.LBFGS,
                               optimizer_params={'lr': 0.1,
                                                 'max_iter': 1,
                                                 'max_eval': 1,
                                                 'history_size': 100
                                                 },
                               ode_solver_kwargs=dict(method='dopri5'),
                               use_adjoint=False
                               )
        nndyn = trainer.train_best(epochs=50,
                                   lr_adapation_rate=0.5,
                                   loss_variance_tolerance=10
                                   )

        evalu = FixedInteractionEvaluator(
            exp_id='linear_small',
            log_dir='./resources/linear',
            n_interactions=200,
            loss_fn=FinalStepMSE(self.x_target, self.total_time),
            ode_solver=odeint,
            ode_solver_kwargs={'method':'dopri5'},
            preserve_intermediate_states=True,
            preserve_intermediate_controls=True,
            preserve_intermediate_times=True,
            preserve_intermediate_energies=True,
            preserve_intermediate_losses=True,
            preserve_params=True
        )

        res = evalu.evaluate(dyn, nndyn.nnc, self.x0, self.total_time, -1)
        evalu.write_to_file(res)
        res = evalu.evaluate(dyn, nndyn.nnc, self.x0, self.total_time, -2)
        evalu.write_to_file(res)

    def test_evaluator_trainer(self):
        dyn = ContinuousTimeInvariantDynamics(self.adjacency_m,
                                              self.driver_m)
        torch.manual_seed(1)
        nn = StackedDenseTimeControl(self.n_nodes, self.n_drivers, n_hidden=1,
                                     hidden_size=self.n_nodes + 4,
                                     activation=torch.nn.functional.elu)
        nndyn = NNCDynamics(dyn, nn)

        evalu = FixedInteractionEvaluator(
            exp_id='linear_small',
            log_dir='./resources/linear',
            n_interactions=200,
            loss_fn=FinalStepMSE(self.x_target, self.total_time),
            ode_solver=odeint,
            ode_solver_kwargs={'method': 'dopri5'},
            preserve_intermediate_states=True,
            preserve_intermediate_controls=True,
            preserve_intermediate_times=True,
            preserve_intermediate_energies=True,
            preserve_intermediate_losses=True,
            preserve_params=True
        )

        trainer = NODECTrainer(nndyn,
                               self.x0,
                               self.x_target,
                               self.total_time,
                               optimizer_class=torch.optim.LBFGS,
                               optimizer_params={'lr': 0.1,
                                                 'max_iter': 1,
                                                 'max_eval': 1,
                                                 'history_size': 100
                                                 },
                               ode_solver_kwargs=dict(method='dopri5'),
                               use_adjoint=False,
                               logger=evalu
                               )

        nndyn = trainer.train_best(epochs=50,
                                   lr_adapation_rate=0.5,
                                   loss_variance_tolerance=10
                                   )



if __name__ == '__main__':
    unittest.main()
