import sys
from importlib.machinery import SourceFileLoader
import importlib.util
import pandas as pd
import gymnasium as gym

# Import TradingEnv
env_spec = importlib.util.spec_from_file_location(
    "environments", "environments.py"
)
environments_module = importlib.util.module_from_spec(env_spec)
sys.modules["environments"] = environments_module
env_spec.loader.exec_module(environments_module)
# Instantiate class here
# module_name.TradingEnv()

# Make environment
trading_days = 252
gym.register(
    id='trading-v0',
    entry_point="environments:TradingEnv",
    max_episode_steps=trading_days
)
df_btc_bitfinex = pd.read_pickle('data/bitfinex2-BTCUSDT-1h.pkl')
df_aapl = pd.read_pickle('data/AAPL_2010_2018.pkl')
env = gym.make(
    'trading-v0', df=df_aapl, 
    portfolio_initial_value=1000, max_episode_duration=100
)
# env.seed(42)

# Import Renderer
rend_spec = importlib.util.spec_from_file_location(
    "renderer", "renderer.py"
)
renderer_module = importlib.util.module_from_spec(rend_spec)
sys.modules["renderer"] = renderer_module
rend_spec.loader.exec_module(renderer_module)

# Load env and run renderer
env.reset()  # Create historical_info in TradingEnv
env.unwrapped.save_for_render(dir="render_logs")
renderer = renderer_module.Renderer(render_logs_dir="render_logs")
renderer.add_line(
    name="sma10", function=lambda df: df["close"].rolling(10).mean(),
    line_options={"width": 1, "color": "purple"}
)
renderer.add_line(
    name="sma20", function=lambda df: df["close"].rolling(20).mean(),
    line_options={"width": 1, "color": "blue"}
)

renderer.run()