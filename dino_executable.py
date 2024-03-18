from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete
import os 
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
class WebGame(Env):
    def __init__(self):
        
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_location = {'top': 300, 'left': 0, 'width': 600, 'height': 500}
        self.done_location = {'top': 405, 'left': 630, 'width': 300, 'height': 70}
        
    def step(self, action):
        
        action_dict = { 0:'space', 1:'down', 2:'no_op' }
        
        if action !=2:
            pydirectinput.press(action_dict[action])
        
        done, done_capt = self.get_done() 
        new_obs = self.get_observation()
        
        reward = 1
        info = {}
        
        return new_obs, reward, done, info
            
    def reset(self):
        
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        
        return self.get_observation()
    
    def get_observation(self):
        
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100,83))
        channel = np.reshape(resized, (1,83,100))
        
        return channel
    
    def get_done(self):
        
        done_cap = np.array(self.cap.grab(self.done_location))
        done_strings = ["G"]
        done = False
        res = pytesseract.image_to_string(done_cap)[:1]
        if res in done_strings:
            done = True

        return done, done_cap

LOG_DIR = "logs"


env = WebGame()
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=1000, learning_starts=0)
model = DQN.load('best_model_90000.zip')
for episode in range(10): 
    observation = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(int(action))
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(episode+1, total_reward))
    time.sleep(1)
