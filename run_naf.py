import pickle
from config import param
from polling import Polling
from algorithms import NAF, sS


def run_env():
    for episode in range(meta_config['n_episodes']):
        observation = env.reset()
        while True:
            env.render()
            meta_action = meta_agent.choose_action(observation)
            agent.target_i, agent.target_q = meta_action
            action = agent.choose_action(observation)
            observation_, reward, end = env.step(action)
            if meta_agent.memory_counter > meta_agent.batch_size * meta_agent.update_per_iter:
                meta_agent.learn()
            meta_agent.store_transition(observation, meta_action, reward, observation_)
            if end:
                meta_agent.record(env.score)
                break
            observation = observation_
        meta_agent.log('{},{},{:.1f},{:.3f},{:.2f}'.format(
            episode + 1, meta_config['n_episodes'], meta_agent.score_his[-1], meta_agent.error_his[-1], meta_agent.epsilon
        ))
    meta_agent.log('---------')
    meta_agent.log('Game over')
    env.destroy()


env = Polling()
meta_config = param(env, 'naf')
if not meta_config['render']:
    env.withdraw()
meta_agent = NAF(**meta_config)
config = param(env, 'sS')
agent = sS(**config)
meta_agent.agent = agent
meta_agent.log('episode,total_episodes,score,error,epsilon')
env.run(run_env)
pickle.dump(meta_agent, open('NAF.p', 'wb'))
meta_agent.report()
