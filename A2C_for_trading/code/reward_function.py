
def backup_reward_function(history):

    portdo_return = (history['portfolio_valuation', -1] / history['portfolio_valuation', 0])
    market_return = (history['data_close', -1] / history['data_close', 0])

    return (portdo_return - market_return)/market_return