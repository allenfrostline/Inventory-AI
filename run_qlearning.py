from config import param
from polling import Polling
from algorithms import QLearning


def run_env():
    for episode in range(config['n_episodes']):
        observation = env.reset()
        while True:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, end = env.step(int(action))
            agent.learn(observation, action, reward, observation_)
            observation = observation_
            if end:
                agent.record(env.score)
                break
        agent.log('{},{},{:.1f},{:.3f},{:.2f}'.format(
            episode + 1, config['n_episodes'], agent.score_his[-1], agent.error_his[-1], agent.epsilon
        ))
    agent.log('---------')
    agent.log('Game over')
    env.destroy()


env = Polling()
config = param(env, 'qlearning')
if not config['render']:
    env.withdraw()
agent = QLearning(**config)
agent.log('episode,total_episodes,score,error,epsilon')
env.run(run_env)
agent.report()
