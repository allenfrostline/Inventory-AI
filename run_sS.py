from config import param
from polling import Polling
from algorithms import sS


def run_env():
    for episode in range(config['n_episodes']):
        observation = env.reset()
        while True:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, end = env.step(action)
            observation = observation_
            if end:
                agent.record(env.score)
                break
        agent.log('{},{},{:.1f},{:.3f}'.format(
            episode + 1, config['n_episodes'], agent.score_his[-1], agent.error_his[-1]
        ))
    agent.log('---------')
    agent.log('Game over')
    env.destroy()


env = Polling()
config = param(env, 'sS')
if not config['render']:
    env.withdraw()
agent = sS(**config)
agent.log('episode,total_episodes,score,error')
env.run(run_env)
agent.log('---------')
agent.log('target q = {:.2f}'.format(config['target_q']))
agent.log('target i = {:.2f}'.format(config['target_i']))
if agent.n_lines == 1:
    agent.log('expected score = {:.2f}'.format(config['expected_score']))
agent.report()
