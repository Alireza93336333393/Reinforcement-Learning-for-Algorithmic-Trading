from typing import Any, Callable, Dict, Optional, Type, Union
import os
import pandas as pd
import pandas_ta as ta
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import gymnasium as gym 
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from alpha_tools import power,rank,ts_argmax,ts_std


    
class PreProcessing:
    def __init__(self):
        pass
    def in_processing(self,df:pd.DataFrame):
        """
        Process the input data by dropping the first two rows, converting relevant columns to numeric,
        setting the index to 'Date'.

        Parameters:
        df : pandas DataFrame
            The input data.

        Returns:
        The processed DataFrame.
        """
        df = df.drop(index=0)
        df = df.drop(index=1)
        df['Date'] = df['Price']
        df = df.drop(columns=['Price'])
        df = df.set_index('Date')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Open'] = pd.to_numeric(df['Open'], errors='coerce')

        return df
    def alpha054(self,o, h, l, c):
        """-(low - close) * power(open, 5) / ((low - high) * power(close, 5))"""
        return (l.sub(c).mul(o.pow(5)).mul(-1)
            .div(l.sub(h).replace(0, -0.0001).mul(c ** 5)))
    def atr(self,df,timeperiod):
        df.ta.atr(length=timeperiod,append=True,prefix='feature_')
    def alpha041(self,h, l, vwap):
        """power(high * low, 0.5 - vwap"""
        return (power(h.mul(l), 0.5)
            .sub(vwap))
    def bb(self,df,length,std):
        df.ta.bbands(length=length,std=std,append=True,prefix='feature_')
    def alpha001(self,c, r):
        """(rank(ts_argmax(power(((returns < 0)
        ? ts_std(returns, 20)
        : close), 2.), 5)) -0.5)"""
        close = c.copy(deep=True)
        return_ = r.copy(deep=True)
        close.loc[return_ < 0] = ts_std(return_, 20)  
        return (rank(ts_argmax(power(close, 2), 5)).mul(-.5))
    def add_ema_sma(self,df:pd.DataFrame):
        """
        Adds the following indicators to the input DataFrame:

        - EMA (lengths 10 and 20)
        - SMA (lengths 10 and 20)

        Parameters:
        df : pandas DataFrame
            The input data.

        Returns:
        The processed DataFrame with added indicators.

        Raises:
        ValueError
            If the input is not a DataFrame.
        """
        try:
            df.ta.ema(length=10, append=True,prefix='feature_')
            df.ta.ema(length=20,append=True,prefix='feature_')
            df.ta.sma(length=10, append=True,prefix='feature_')
            df.ta.sma(length=20,append=True,prefix='feature_')
            return df
        except ValueError:
            print(f'make sure you pass a DataFrame:{type(df)}') 
    def gann_high_low1(self,df, HPeriod=15, LPeriod=21):
        """
        Calculates the Gann High-Low indicator.

        Parameters:
        df : pandas DataFrame
            The input data. Must contain the 'High', 'Low', and 'Close' columns.
        HPeriod : int, optional
            The length of the short Simple Moving Average (SMA) used in the calculation of the Gann
            High-Low indicator. Default is 15.
        LPeriod : int, optional
            The length of the long SMA used in the calculation of the Gann High-Low indicator. Default
            is 21.

        Returns:
        A DataFrame with the Gann High-Low indicator added as a new column.

        Notes:
        The Gann High-Low indicator is a technical analysis indicator that helps identify trend changes
        in a security's price. It is calculated as the shorter of the two SMAs, or the longer of the two
        SMAs, depending on whether the security is in an uptrend or a downtrend.
        """
        sma_1 = df['High'].rolling(window=HPeriod).mean()
        sma_2 = df['Low'].rolling(window=LPeriod).mean()
        # Calculate HLd
        HLd = np.where(df['Close'] > sma_1.shift(1), 1, 
                   np.where(df['Close'] < sma_2.shift(1), -1, 0))
        # Calculate HLv
        HLv = np.where(HLd != 0, HLd, 0)
        # Calculate HiLo
        HiLo = np.where(HLv == -1, sma_1, sma_2)
        df['feature_gann_hiLo'] = HiLo

        return df
    def add_return(self,df):
        df.ta.percent_return(cumulative=True,append=True)

def make_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    df:pd.DataFrame,
    positions:list[int] = [0,1,-1],
    reward_function=None,
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
        Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
        if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
        See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                import gymnasium as gym 
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "logs"}
                kwargs.update(env_kwargs)
                try:
                    if reward_function:
                        env = gym.make(env_id,
                                       df=df,
                                       positions=positions,
                                       reward_function=reward_function,**kwargs)  # type: ignore[arg-type]
                    else : 
                        env = gym.make(env_id,
                                       df=df,
                                       positions=positions,**kwargs)  # type: ignore[arg-type]
                except TypeError:
                    if reward_function:
                        env = gym.make(env_id,
                                       df=df,
                                       positions=positions,
                                       reward_function=reward_function,**kwargs)  # type: ignore[arg-type]
                    else : 
                        env = gym.make(env_id,
                                       df=df,
                                       positions=positions,**kwargs)  # type: ignore[arg-type]
            else:
                env = env_id(**env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env

def for_alpha(df):
    o = df.open
    h = df.high
    l = df.low
    c = df.close
    v = df.volume
    vwap = o.add(h).add(l).add(c).div(4)
    adv20 = v.rolling(20).mean()
    r = df.cumpctret_1
    return o,h,l,c,v,vwap,adv20,r


