import numpy as np
from gym import spaces
from polling import MAX_STEPS, MAX_I, MAX_Q, UTILIZATION, K_II, K_0I, C_HOLDING, C_BACKLOGGING, LONGTERM_BONUS


# agent configuration
def param(env, name):
    n_lines = env.state_space.shape[1]
    if n_lines == 1:
        if name == 'sS': return {
            'target_q': np.sqrt(2 * K_0I * UTILIZATION * (1 - UTILIZATION) * C_HOLDING / (C_HOLDING + C_BACKLOGGING) / C_BACKLOGGING),
            'target_i': np.sqrt(2 * K_0I * UTILIZATION * (1 - UTILIZATION) * C_BACKLOGGING / (C_HOLDING + C_BACKLOGGING) / C_HOLDING),
            'n_lines': n_lines,
            'expected_score': (LONGTERM_BONUS / 100 - np.sqrt(2 * K_0I * UTILIZATION * (1 - UTILIZATION) * C_HOLDING * C_BACKLOGGING / (C_HOLDING + C_BACKLOGGING)) + K_II * UTILIZATION) * MAX_STEPS,
            'utilization': UTILIZATION,
            'n_episodes': 5000,
            'plot': True,
            'render': False,
        }
        if name == 'qlearning': return {
            'action_space': env.action_space,
            'state_space': env.state_space,
            'learning_rate': 0.01,
            'gamma': 0.99,
            # 'epsilon': 0.3,
            'epsilon_min': 0.1,
            'epsilon_max': 1.0,
            'epsilon_deflator': 0.9995,
            'utilization': UTILIZATION,
            'n_episodes': 5000,
            'plot': True,
            'render': False,
        }
        if name == 'dqn': return {
            'action_space': env.action_space,
            'state_space': env.state_space,
            'hidden_neurons': 64,
            'learning_rate': 0.01,
            'gamma': 0.99,
            # 'epsilon': 0.3,
            'epsilon_min': 0.05,
            'epsilon_max': 1.00,
            'epsilon_deflator': 0.9994,
            'replace_target_iter': 200,
            'memory_size': 10000,
            'batch_size': 100,
            'utilization': UTILIZATION,
            'n_episodes': 5000,
            'plot': True,
            'render': False,
        }
        if name == 'naf': return {
            'action_space': spaces.Box(low=np.array([0, 0]), high=np.array([MAX_I, MAX_Q]), dtype=np.float32),
            'state_space': env.state_space,
            'hidden_neurons': 200,
            'learning_rate': 0.0001,
            'gamma': 0.99,
            # 'epsilon': 0.3,
            'epsilon_min': 0.3,
            'epsilon_max': 0.7,
            'epsilon_deflator': 0.9991,
            'update_per_iter': 1,
            'memory_size': 10000,
            'batch_size': 50,
            'tau': 0.001,
            'utilization': UTILIZATION,
            'n_episodes': 5000,
            'plot': True,
            'render': False,
        }
    if n_lines == 2:
        if name == 'sS': return {
            'target_q': np.sqrt(2 * K_0I * UTILIZATION * (1 - UTILIZATION) * C_HOLDING / (C_HOLDING + C_BACKLOGGING) / C_BACKLOGGING),
            'target_i': np.sqrt(2 * K_0I * UTILIZATION * (1 - UTILIZATION) * C_BACKLOGGING / (C_HOLDING + C_BACKLOGGING) / C_HOLDING),
            'n_lines': n_lines,
            'utilization': UTILIZATION,
            'n_episodes': 500,
            'plot': True,
            'state_space': env.state_space,
            'render': False,
        }
        if name == 'qlearning': return {
            'action_space': env.action_space,
            'state_space': env.state_space,
            'learning_rate': 0.01,
            'gamma': 0.99,
            # 'epsilon': 0.3,
            'epsilon_min': 0.1,
            'epsilon_max': 1.0,
            'epsilon_deflator': 0.9995,
            'utilization': UTILIZATION,
            'n_episodes': 5000,
            'plot': True,
            'render': False,
        }
        if name == 'dqn': return {
            'action_space': env.action_space,
            'state_space': env.state_space,
            'hidden_neurons': 64,
            'learning_rate': 0.02,
            'gamma': 0.99,
            # 'epsilon': 0.3,
            'epsilon_min': 0.1,
            'epsilon_max': 1.0,
            'epsilon_deflator': 0.9995,
            'replace_target_iter': 200,
            'memory_size': 10000,
            'batch_size': 100,
            'utilization': UTILIZATION,
            'n_episodes': 5000,
            'plot': True,
            'render': False,
        }
        if name == 'naf': return {
            'action_space': spaces.Box(low=np.array([0, 0]), high=np.array([MAX_I, MAX_Q]), dtype=np.float32),
            'state_space': env.state_space,
            'hidden_neurons': 100,
            'learning_rate': 0.001,
            'gamma': 0.99,
            # 'epsilon': 0.3,
            'epsilon_min': 0.3,
            'epsilon_max': 0.9,
            'epsilon_deflator': 0.999,
            'update_per_iter': 1,
            'memory_size': 5000,
            'batch_size': 100,
            'tau': 0.01,
            'utilization': UTILIZATION,
            'n_episodes': 5000,
            'plot': True,
            'render': False,
        }
    if n_lines == 5:
        if name == 'sS': return {
            'target_q': np.sqrt(2 * K_0I * UTILIZATION * (1 - UTILIZATION) * C_HOLDING / (C_HOLDING + C_BACKLOGGING) / C_BACKLOGGING),
            'target_i': np.sqrt(2 * K_0I * UTILIZATION * (1 - UTILIZATION) * C_BACKLOGGING / (C_HOLDING + C_BACKLOGGING) / C_HOLDING),
            'n_lines': n_lines,
            'utilization': UTILIZATION,
            'n_episodes': 5000,
            'plot': True,
            'render': False,
        }
        if name == 'qlearning': return {
            'action_space': env.action_space,
            'state_space': env.state_space,
            'learning_rate': 0.01,
            'gamma': 0.99,
            # 'epsilon': 0.3,
            'epsilon_min': 0.1,
            'epsilon_max': 1.0,
            'epsilon_deflator': 0.9995,
            'utilization': UTILIZATION,
            'n_episodes': 5000,
            'plot': True,
            'render': False,
        }
        if name == 'dqn': return {
            'action_space': env.action_space,
            'state_space': env.state_space,
            'hidden_neurons': 64,
            'learning_rate': 0.005,
            'gamma': 0.9,
            # 'epsilon': 0,
            'epsilon_min': 0.05,
            'epsilon_max': 1.00,
            'epsilon_deflator': 0.9994,
            'replace_target_iter': 100,
            'memory_size': 5000,
            'batch_size': 100,
            'utilization': UTILIZATION,
            'n_episodes': 500,
            'plot': True,
            'render': False,
        }
        if name == 'naf': return {
            'action_space': spaces.Box(low=np.array([0, 0]), high=np.array([MAX_I, MAX_Q]), dtype=np.float32),
            'state_space': env.state_space,
            'hidden_neurons': 200,
            'learning_rate': 0.0005,
            'gamma': 0.99,
            # 'epsilon': 0.3,
            'epsilon_min': 0.3,
            'epsilon_max': 0.9,
            'epsilon_deflator': 0.999,
            'update_per_iter': 1,
            'memory_size': 5000,
            'batch_size': 100,
            'tau': 0.01,
            'utilization': UTILIZATION,
            'n_episodes': 5000,
            'plot': True,
            'render': False,
        }