# A2C_for_trading


## project
this project is useing RL to trade a asssest in a market . i build it for learning purpespese so learning by doing 

## how it work 
i think the only algo in the AI world that reperesent human chractestic is RL so i though why not use it doe trading ? 

im not a qount so i use public alphas which you cant called them **ALPHAS**. 
i use these public alphas [ATR,BB] and this papre for more ['public alphas'](https://arxiv.org/pdf/1601.00991) 

the reward function was created so it rewards the agent if it outperforms the benchmarke and also has good pnl but it dosent do exacly this
the agent learns to hack the reward function (or this is what i guasse!!) and then use the weakness of the reward function to get more reward .
the weeakness of the reward function is that it trys to limit the capeblitit of the agent and also limit the beivere of the market to only two things,

* that the market can be outperform only by given a sets of reward for outperforming the market by doing anything ot can
* that using only pnl can explain how good a stratgy is and how good a behiver is

these resoinig can lead to this:

![alt text](https://github.com/Alireza93336333393/A2C_for_trading/blob/main/Screenshot%20from%202025-03-20%2010-34-50%20(1).png)

the agent takes too much bad action that the env is done befor it starts becuse it takes bad action the mony is overe . 
then it start to form the behivere that buying and holding is a good thing (which is what i exacly didint want) becuse the reward function rewards this kind of behiver it tris to be better at it . 

here you can see what im talking about:

![alt text](https://github.com/Alireza93336333393/A2C_for_trading/blob/main/Screenshot%20from%202025-03-20%2010-34-50.png)

![alt text](https://github.com/Alireza93336333393/A2C_for_trading/blob/main/Screenshot%20from%202025-03-21%2008-30-10.png)

![alt text]()

## TODO 
