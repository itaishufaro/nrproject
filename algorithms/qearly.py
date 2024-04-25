import logging
import numpy as np

import gym.spaces as spaces
from rlberry.agents import IncrementalAgent
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.utils.writers import PeriodicWriter

logger = logging.getLogger(__name__)


class QEarlyAgent(IncrementalAgent):
    """
    Implementation of the QEarlySetteled algorithm.

    Parameters
    ----------
    env : gym.Env
        Environment with discrete states and actions.
    n_episodes : int
        Number of episodes
    gamma : double, default: 1.0
        Discount factor in [0, 1].
    horizon : int
        Horizon of the objective function.
    bonus_scale_factor : double, default: 1.0
        Constant by which to multiply the exploration bonus, controls
        the level of exploration.
    bonus_type : {"simplified_bernstein"}
        Type of exploration bonus. Currently, only "simplified_bernstein"
        is implemented.
    """
    name = "QEarly"

    def __init__(self,
                 env,
                 n_episodes=1000,
                 horizon=100,
                 bonus_scale_factor=1.0,
                 p=0.001,
                 debug=False,
                 **kwargs):
        # init base class
        IncrementalAgent.__init__(self, env, **kwargs)

        self.n_episodes = n_episodes
        self.horizon = horizon
        self.bonus_scale_factor = bonus_scale_factor
        self.debug = debug
        self.p = p

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf or r_range == 0.0:
            logger.warning("{}: Reward range is  zero or infinity. ".format(self.name)
                           + "Setting it to 1.")
            r_range = 1.0


        # initialize
        self.reset()

    def reset(self, **kwargs):
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n

        self.episode = 0
        self._rewards = np.zeros(self.n_episodes)
        self.counter = DiscreteCounter(self.env.observation_space,
                                       self.env.action_space)

        # Q values
        self.Q = np.zeros((H, S, A)) + H
        self.Q_ucb = np.zeros((H, S, A)) + H
        self.Q_R = np.zeros((H, S, A)) + H
        self.V = np.zeros((H+1, S))
        self.V[:-1, :] = H
        self.V_R = np.zeros((H+1, S)) + H
        self.V_R[:-1, :] = H
        self.Q_lcb = np.zeros((H, S, A))
        self.V_lcb = np.zeros((H+1, S))

        # Counter
        self.N = np.zeros((H, S, A))

        # moments
        self.mu_ref = np.zeros((H, S, A))
        self.sigma_ref = np.zeros((H, S, A))
        self.mu_adv = np.zeros((H, S, A))
        self.sigma_adv = np.zeros((H, S, A))
        self.delta_R = np.zeros((H, S, A))
        self.B_R = np.zeros((H, S, A))
        self.u_ref = np.ones((S,1))


        # default writer
        self.writer = PeriodicWriter(self.name,
                                     log_every=5 * logger.getEffectiveLevel())

    def update_bonus(self, h, state, action, T):
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n
        n = self.N[h, state, action]
        if n == 0:
            Bh_next = H
        else:
            Bh_next = self.bonus_scale_factor * np.sqrt(np.log((S*A*T / self.p))/n) * (
                np.sqrt(np.abs(self.sigma_ref[h, state, action] - (self.mu_ref[h, state, action]) ** 2))
                + np.sqrt(H) * np.sqrt(np.abs(self.sigma_adv[h, state, action] - (self.mu_adv[h, state, action]) ** 2))
            )
        self.delta_R[h, state, action] = Bh_next - self.B_R[h, state, action]
        self.B_R[h, state, action] = Bh_next

    def update_moments(self, h, state, action, next_state):
        H = self.horizon
        n = self.N[h, state, action]
        etan = (H+1)/(H+n)
        if n != 0:
            self.mu_ref[h, state, action] = ((1 - 1/n)*self.mu_ref[h, state, action]
                                             + 1/n * self.V_R[h+1, next_state])
            self.sigma_ref[h, state, action] = ((1 - 1/n)*self.sigma_ref[h, state, action]
                                                + 1/n * (self.V_R[h+1, next_state]) ** 2)
            self.mu_adv[h, state, action] = ((1-etan) * self.mu_adv[h, state, action] +
                                             etan * (self.V[h+1, next_state] - self.V_R[h+1, next_state]))
            self.sigma_adv[h, state, action] = ((1-etan) * self.sigma_adv[h, state, action] +
                                                  etan * (self.V[h+1, next_state] - self.V_R[h+1, next_state]) ** 2)

    def update_ucb_q(self, h, state, action, next_state, reward):
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n
        T = self.episode + 1
        n = self.N[h, state, action]
        etan = (H+1)/(H+n)
        if n == 0:
            b = H
        else:
            b = self.bonus_scale_factor * np.sqrt((H ** 3) * np.log(S * A * T / self.p) / n)
        self.Q_ucb[h, state, action] = (1 - etan) * self.Q_ucb[h, state, action] + etan * (
                reward + self.V[h+1, next_state] + b)

    def update_lcb_q(self, h, state, action, next_state, reward):
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n
        T = self.episode + 1
        n = self.N[h, state, action]
        etan = (H + 1) / (H + n)
        if n == 0:
            b = H
        else:
            b = self.bonus_scale_factor * np.sqrt((H ** 3) * np.log(S * A * T / self.p) / n)
        self.Q_lcb[h, state, action] = (1 - etan) * self.Q_lcb[h, state, action] + etan * (
                reward + self.V_lcb[h + 1, next_state] + b)

    def update_ucb_q_advantage(self, h, state, action, next_state, reward):
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n
        T = self.episode + 1
        n = self.N[h, state, action]
        etan = (H + 1) / (H + n)
        self.update_moments(h, state, action, next_state)
        self.update_bonus(h, state, action, self.episode)
        if n == 0:
            b = H
        else:
            b = (self.B_R[h, state, action] + (1-etan) * self.delta_R[h, state, action] / etan
                 + self.bonus_scale_factor * (np.power(H, 2) * np.log(S*A*T/self.p))/(np.power(n, 0.75)))
        self.Q_R[h, state, action] = (1 - etan) * self.Q_R[h, state, action] + etan * (
                reward + self.V[h + 1, next_state] - self.V_R[h+1, next_state] +
                self.mu_ref[h, state, action] + b)

    def policy(self, state, hh=0, **kwargs):
        """ Recommended policy. """
        return self.Q[hh, state, :].argmax()

    def _get_action(self, state, hh=0):
        """ Sampling policy. """
        return self.Q[hh, state, :].argmax()

    def _update(self, state, action, next_state, reward, hh):
        self.N[hh, state, action] += 1
        self.update_ucb_q(hh, state, action, next_state, reward)
        self.update_lcb_q(hh, state, action, next_state, reward)
        self.update_ucb_q_advantage(hh, state, action, next_state, reward)
        self.Q[hh, state, action] = np.min(np.array([
            self.Q_ucb[hh, state, action],
            self.Q_lcb[hh, state, action],
            self.Q_R[hh, state, action]
        ]))
        self.V[hh, state] = np.max(self.Q[hh, state, :])
        self.V_lcb[hh, state] = np.max(np.array([
            np.max(self.Q_lcb[hh, state, :]),
            self.V_lcb[hh, state]
        ]))
        if self.V[hh, state] - self.V_lcb[hh, state] > 1:
            self.V_R[hh, state] = self.V[hh, state]
            self.u_ref[state] = 1
        elif self.u_ref[state] == 1:
            self.V_R[hh, state] = self.V[hh, state]
            self.u_ref[state] = 0

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        for hh in range(self.horizon):
            action = self._get_action(state, hh)
            next_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward  # used for logging only

            self.counter.update(state, action)

            self._update(state, action, next_state, reward, hh)

            state = next_state
            if done:
                break

        # update info
        ep = self.episode
        self._rewards[ep] = episode_rewards
        self.episode += 1

        # writer
        if self.writer is not None:
            self.writer.add_scalar("ep reward", episode_rewards, self.episode)
            self.writer.add_scalar("total reward", self._rewards[:ep].sum(), self.episode)
            self.writer.add_scalar("n_visited_states", self.counter.get_n_visited_states(), self.episode)

        # return sum of rewards collected in the episode
        return episode_rewards

    def partial_fit(self, fraction, **kwargs):
        assert 0.0 < fraction <= 1.0
        n_episodes_to_run = int(np.ceil(fraction * self.n_episodes))
        count = 0
        while count < n_episodes_to_run and self.episode < self.n_episodes:
            self._run_episode()
            count += 1

        info = {"n_episodes": self.episode,
                "episode_rewards": self._rewards[:self.episode]}

        return info
