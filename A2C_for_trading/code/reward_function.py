
def backup_reward_function(history):

    portfolio_return = (history['portfolio_valuation', -1] / history['portfolio_valuation', 0])
    market_return = (history['data_close', -1] / history['data_close', 0])

    return (portfolio_return - market_return)/market_return
