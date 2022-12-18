import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import seals

env = gym.make('seals/Pendulum-v0')
expert = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
    tensorboard_log="./tb_log/",
)
expert.learn(
    total_timesteps=100,
) 
expert.save('./model/expert')

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    make_vec_env(
        "seals/Pendulum-v0",
        n_envs=5,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
    ),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

import gym


venv = make_vec_env("seals/Pendulum-v0", n_envs=8, rng=rng)

learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.001,
    learning_rate=0.0003,
    n_epochs=10,
    tensorboard_log="./tb_log/",
)
reward_net = BasicRewardNet(
    venv.observation_space,
    venv.action_space,
    use_action=False,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

expert_rewards, _ = evaluate_policy(
    expert, env, 100, return_episode_rewards=True
)
   

gail_trainer.train(30000)  # Note: set to 300000 for better results
learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)
learner.save('./model/gail')

del learner, reward_net, gail_trainer

learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=-0.001,
    learning_rate=0.0003,
    n_epochs=10,
    tensorboard_log="./tb_log/",
)
reward_net = BasicRewardNet(
    venv.observation_space,
    venv.action_space,
    use_action=False,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

gail_trainer.train(30000)  # Note: set to 300000 for better results

learner.save('./model/gail_neg')

learner_rewards_after_training_neg_ent, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)

import matplotlib.pyplot as plt
import numpy as np

print(np.mean(expert_rewards))
print(np.mean(learner_rewards_after_training))
print(np.mean(learner_rewards_after_training_neg_ent))

plt.hist(
    [
        expert_rewards, 
        learner_rewards_after_training,
        learner_rewards_after_training_neg_ent,
    ],
    label=["expert", "trained", "trained_neg_ent"],
)
plt.legend()
plt.show()

# import imageio

# images = []
# obs = learner.env.reset()
# img = learner.env.render(mode="rgb_array")
# for i in range(350):
#     images.append(img)
#     action, _ = learner.predict(obs)
#     obs, _, _ ,_ = learner.env.step(action)
#     img = learner.env.render(mode="rgb_array")

# imageio.mimsave("gail.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
