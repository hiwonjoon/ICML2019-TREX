import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import gym
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from imgcat import imgcat

from tf_commons.ops import *
from siamese_ranker import PPO2Agent

class Policy(object):
    def __init__(self,ob_dim,ac_dim,embedding_dims=512):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.inp = tf.placeholder(tf.float32,[None,ob_dim])
            self.l = tf.placeholder(tf.float32,[None,ac_dim])
            self.l2_reg = tf.placeholder(tf.float32,[])

            with tf.variable_scope('weights') as param_scope:
                self.fc1 = Linear('fc1',ob_dim,embedding_dims)
                self.fc2 = Linear('fc2',embedding_dims,embedding_dims)
                self.fc3 = Linear('fc3',embedding_dims,embedding_dims)
                self.fc4 = Linear('fc4',embedding_dims,embedding_dims)
                self.fc5 = Linear('fc5',embedding_dims,ac_dim)

            self.param_scope = param_scope

            # build graph
            def _policy(x):
                _ = tf.nn.relu(self.fc1(x))
                _ = tf.nn.relu(self.fc2(_))
                _ = tf.nn.relu(self.fc3(_))
                _ = tf.nn.relu(self.fc4(_))
                r = self.fc5(_)
                return r

            self.ac = _policy(self.inp)

            loss = tf.reduce_sum((self.ac-self.l)**2,axis=1)
            self.loss = tf.reduce_mean(loss,axis=0)

            weight_decay = tf.reduce_sum(self.fc1.w**2) + tf.reduce_sum(self.fc2.w**2) + tf.reduce_sum(self.fc3.w**2)
            self.l2_loss = self.l2_reg * weight_decay

            self.optim = tf.train.AdamOptimizer(1e-4)
            self.update_op = self.optim.minimize(self.loss+self.l2_loss,var_list=self.parameters(train=True))

            self.saver = tf.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)

            ################ Miscellaneous
            self.init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

        self.sess.run(self.init_op)

    def parameters(self,train=False):
        if train:
            return tf.trainable_variables(self.param_scope.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

    def train(self,D,batch_size=64,iter=20000,l2_reg=0.001,debug=False):
        sess = self.sess

        obs,acs,_ = D

        idxes = np.random.permutation(len(obs))
        train_idxes = idxes[:int(len(obs)*0.8)]
        valid_idxes = idxes[int(len(obs)*0.8):]

        def _batch(idx_list):
            batch = []

            if len(idx_list) > batch_size:
                idxes = np.random.choice(idx_list,batch_size,replace=False)
            else:
                idxes = idx_list

            for i in idxes:
                batch.append((obs[i],acs[i]))

            b_ob,b_ac = zip(*batch)
            b_ob,b_ac = np.array(b_ob),np.array(b_ac)

            return b_ob,b_ac

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_ob,b_ac = _batch(train_idxes)

            with self.graph.as_default():
                loss,l2_loss,_ = sess.run([self.loss,self.l2_loss,self.update_op],feed_dict={
                    self.inp:b_ob,
                    self.l:b_ac,
                    self.l2_reg:l2_reg,
                })

                b_ob,b_ac = _batch(valid_idxes)
                valid_loss= sess.run(self.loss,feed_dict={
                    self.inp:b_ob,
                    self.l:b_ac,
                    self.l2_reg:l2_reg,
                })

            if debug:
                if it % 100 == 0 or it < 10:
                    tqdm.write(('loss: %f (l2_loss: %f), valid_loss: %f'%(loss,l2_loss,valid_loss)))

            #if valid_acc >= 0.95:
            #    print('loss: %f (l2_loss: %f), acc: %f, valid_acc: %f'%(loss,l2_loss,acc,valid_acc))
            #    print('early termination@%08d'%it)
            #    break

    def act(self, observation, reward, done):
        sess = self.sess

        with self.graph.as_default():
            ac = sess.run(self.ac,feed_dict={self.inp:observation[None]})[0]

        return ac

class Dataset(object):
    def __init__(self,env):
        self.env = env

    def gen_traj(self,agent,min_length):
        obs, actions, rewards = [self.env.reset()], [], []

        # For debug purpose
        last_episode_idx = 0
        acc_rewards = []

        while True:
            action = agent.act(obs[-1], None, None)
            ob, reward, done, _ = self.env.step(action)

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)

            if done:
                if len(obs) < min_length:
                    obs.pop()
                    obs.append(self.env.reset())

                    acc_rewards.append(np.sum(rewards[last_episode_idx:]))
                    last_episode_idx = len(rewards)
                else:
                    obs.pop()

                    acc_rewards.append(np.sum(rewards[last_episode_idx:]))
                    last_episode_idx = len(rewards)
                    break

        return np.concatenate(obs,axis=0), np.concatenate(actions,axis=0), np.concatenate(rewards,axis=0), np.mean(acc_rewards)

    def prebuilt(self,agents,min_length):
        assert len(agents)>0, 'no agent given'
        trajs = []
        for agent in tqdm(agents):
            (*traj), avg_acc_reward = self.gen_traj(agent,min_length)

            trajs.append(traj)
            tqdm.write('model: %s avg reward: %f'%(agent.model_path,avg_acc_reward))
        obs,actions,rewards = zip(*trajs)
        self.trajs = (np.concatenate(obs,axis=0),np.concatenate(actions,axis=0),np.concatenate(rewards,axis=0))

        print(self.trajs[0].shape,self.trajs[1].shape,self.trajs[2].shape)

def train(args):
    logdir = Path(args.logbase_path) / args.env_id
    if logdir.exists() :
        c = input('log is already exist. continue [Y/etc]? ')
        if c in ['YES','yes','Y']:
            import shutil
            shutil.rmtree(str(logdir))
        else:
            print('good bye')
            exit()
    logdir.mkdir(parents=True)
    logdir = str(logdir)
    env = gym.make(args.env_id)

    agent = PPO2Agent(env,args.env_type,str(args.learner_path))

    dataset = Dataset(env)
    dataset.prebuilt([agent],args.min_length)

    policy = Policy(env.observation_space.shape[0],env.action_space.shape[0])

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)

    D = dataset.trajs
    policy.train(D,l2_reg=args.l2_reg,debug=True)
    policy.saver.save(sess,logdir+'/model.ckpt',write_meta_graph=False)

def eval(args):
    env = gym.make(args.env_id)

    logdir = str(Path(args.logbase_path) / args.env_id)
    policy = Policy(env.observation_space.shape[0],env.action_space.shape[0])

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)
    policy.saver.restore(sess,logdir+'/model.ckpt')

    from performance_checker import gen_traj
    from gym.wrappers import Monitor
    perfs = []
    for j in range(args.num_trajs):
        if j == 0 and args.video_record:
            wrapped = Monitor(env, './video/',force=True)
        else:
            wrapped = env

        perfs.append(gen_traj(wrapped,policy,args.render,args.max_len))
    print(logdir, ',', np.mean(perfs), ',', np.std(perfs))


if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--min_length', default=1000,type=int, help='minimum length of trajectory generated by each agent')
    parser.add_argument('--learner_path', default='', help='path of learning agents')
    parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
    parser.add_argument('--logbase_path', default='./learner/models_bc/', help='path to log base (env_id will be concatenated at the end)')
    parser.add_argument('--eval', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    parser.add_argument('--max_len', default=1000, type=int)
    parser.add_argument('--num_trajs', default=10, type=int, help='path of learning agents')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--video_record', action='store_true')
    args = parser.parse_args()

    if not args.eval :
        train(args)
    else:
        eval(args)
