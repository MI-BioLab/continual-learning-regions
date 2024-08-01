from setting_strategies import SingleStreamStrategy, MultiStreamStrategy

def get_setting_strategy(value,
                         model, 
                         **kwargs):
    match value:
        case 0:
            return SingleStreamStrategy(model, **kwargs)
        case 1:
            return MultiStreamStrategy(model, **kwargs)
    raise ValueError("get_setting_strategy: invalid setting strategy") 
    
__all__ = ["get_setting_strategy"]