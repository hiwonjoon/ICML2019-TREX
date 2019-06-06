def normalize_state(obs):
    return obs / 255.0


#custom masking function for covering up the score/life portions of atari games
def mask_score(obs, env_name):
    obs_copy = obs.copy()
    if env_name == "spaceinvaders" or env_name == "breakout" or env_name == "pong":
        #takes a stack of four observations and blacks out (sets to zero) top n rows
        n = 10
        #no_score_obs = copy.deepcopy(obs)
        obs_copy[:,:n,:,:] = 0
    elif env_name == "beamrider":
        n_top = 16
        n_bottom = 11
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name == "enduro":
        n_top = 0
        n_bottom = 14
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
        #cuts out place in race, but keeps odometer
    elif env_name == "hero":
        n_top = 0
        n_bottom = 30
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name == "qbert":
        n_top = 12
        #n_bottom = 0
        obs_copy[:,:n_top,:,:] = 0
        #obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name == "seaquest":
        n_top = 12
        n_bottom = 16
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
        #cuts out divers and oxygen
    elif env_name == "mspacman":
        n_bottom = 15 #mask score and number lives left
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name == "videopinball":
        n_top = 15
        obs_copy[:,:n_top,:,:] = 0
    elif env_name == "montezumarevenge":
        n_top = 10
        obs_copy[:,:n_top,:,:] = 0
    else:
        print("NOT MASKING SCORE FOR GAME: " + env_name)
        pass
        #n = 20
        #obs_copy[:,-n:,:,:] = 0
    return obs_copy

def preprocess(ob, env_name):
    #print("masking on env", env_name)
    return mask_score(normalize_state(ob), env_name)
