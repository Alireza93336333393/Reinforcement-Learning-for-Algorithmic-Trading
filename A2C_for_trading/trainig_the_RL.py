#____________________data part_____________________#
import pandas as pd  
from usefull import PreProcessing,for_alpha


path = '/home/alireza/Desktop/alireza/jupyter_sttuf/bitcoin.csv'


df = pd.read_csv(path)
prep = PreProcessing()

df = prep.in_processing(df)

prep.add_return(df)
df.columns = df.columns.str.lower()
o,h,l,c,v,vwap,adv20,r = for_alpha(df)

df['feature_alpha001'] = prep.alpha001(c,r)# this is ok the env went crazy becuse the agent takes bad action and end the monye
df['feature_alpha041'] = prep.alpha041(h,l,vwap)#this is okay with the env !!
df['feature_alpha054'] = prep.alpha054(o,h,l,c)#this is also okay with the env you can use it 
prep.atr(df,20)
prep.bb(df,length=20,std=2)
df.fillna(0,inplace=True)
df.columns = df.columns.str.lower()
df.drop(columns=['cumpctret_1'],inplace=True)
#_____________________train part_____________________#
from stable_baselines3.a2c import A2C
import gymnasium as gym
import gym_trading_env
from reward_function import backup_reward_function
from action_record import ActionRecorderCallback

action_path = 'your_path.csv'

env = gym.make("TradingEnv", df = df,positions = [-1,0,1],trading_fees=0.02,reward_function=backup_reward_function)

model = A2C("MlpPolicy",env,device='cpu',tensorboard_log='./A2C_pnl+outbench')


model.learn(300_000,callback=ActionRecorderCallback(action_path=action_path)) 

model.save(path='/home/alireza/Desktop/alireza/jupyter_sttuf/A2C_pnl+oubench_reward_function')