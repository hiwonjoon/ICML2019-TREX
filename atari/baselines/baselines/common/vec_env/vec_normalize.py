from . import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, eval=False):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.eval = eval

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            if not self.eval :
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            if not self.eval :
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def save(self,loc):
        s = {}
        if self.ret_rms :
            s['ret_rms'] = self.ret_rms
        if self.ob_rms :
            s['ob_rms'] = self.ob_rms

        import pickle
        with open(loc+'.env_stat.pkl', 'wb') as f :
            pickle.dump(s,f)

    def load(self,loc):
        import pickle
        with open(loc+'.env_stat.pkl', 'rb') as f :
            s = pickle.load(f)

        if self.ret_rms :
            self.ret_rms = s['ret_rms']
        if self.ob_rms :
            self.ob_rms = s['ob_rms']


class VecNormalizeRewards(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, cliprew=10., gamma=0.99, epsilon=1e-8, eval=False):
        VecEnvWrapper.__init__(self, venv)
        self.ret_rms = RunningMeanStd(shape=())
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.eval = eval

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        if self.ret_rms:
            if not self.eval :
                self.ret_rms.update(self.ret)
            rews = np.clip((rews - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos


    def reset(self):
        print("Env resetting!!!!!!!!!!!!!!")
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return obs

    def save(self,loc):
        s = {}
        if self.ret_rms :
            s['ret_rms'] = self.ret_rms

        import pickle
        with open(loc+'.env_stat.pkl', 'wb') as f :
            pickle.dump(s,f)

    def load(self,loc):
        import pickle
        with open(loc+'.env_stat.pkl', 'rb') as f :
            s = pickle.load(f)

        if self.ret_rms :
            self.ret_rms = s['ret_rms']
        
