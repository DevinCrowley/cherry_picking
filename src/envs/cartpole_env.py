import numpy as np
import torch


class Cartpole_Env:

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag_max = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_size = 1
        self.observation_size = 4

        self.state = None

        self.steps_beyond_terminated = None

    def set_state(self, state):
        (x, x_dot, theta, theta_dot) = state
        self.state = (x, x_dot, theta, theta_dot)
        self.validate_state()

    def validate_state(self):
        if self.state is not None:
            state = np.zeros(len(self.state))
            for i, s in enumerate(self.state):
                if hasattr(s, '__len__'):
                    state[i] = s.item()
                else:
                    state[i] = s
        self.state = state

    def step(self, action):
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag_max * action
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)
        self.validate_state()

        terminated = self.compute_done()
        reward = self.compute_reward(action)
        info = {}

        return np.array(self.state, dtype=np.float32), reward, terminated, info
    
    def compute_done(self):
        x, x_dot, theta, theta_dot = self.state
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        return terminated

    def compute_reward(self, action):
        terminated = self.compute_done()
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                print(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

    def reset(self, state=None):
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = -0.05, 0.05
        self.steps_beyond_terminated = None
        if state is None:
            self.state = np.random.uniform(low=low, high=high, size=(4,))
        else:
            self.set_state(state)

        return np.array(self.state, dtype=np.float32)