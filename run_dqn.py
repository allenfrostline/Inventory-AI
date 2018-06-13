from config import param
from polling import Polling
from algorithms import DQN


def run_env():
    for episode in range(config['n_episodes']):
        observation = env.reset()
        while True:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, end = env.step(int(action))
            agent.store_transition(observation, action, reward, observation_)
            if agent.memory_counter > agent.batch_size * 5:
                agent.learn()
            if end:
                agent.record(env.score)
                break
            observation = observation_
        agent.log('{},{},{:.1f},{:.3f},{:.2f}'.format(
            episode + 1, config['n_episodes'], agent.score_his[-1], agent.error_his[-1], agent.epsilon
        ))
    agent.log('---------')
    agent.log('Game over')
    env.destroy()


env = Polling()
config = param(env, 'dqn')
if not config['render']:
    env.withdraw()
agent = DQN(**config)
agent.log('episode,total_episodes,score,error,epsilon')
env.run(run_env)
agent.report()
