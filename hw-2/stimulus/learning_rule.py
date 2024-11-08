import  numpy as np

def zero_model_create(num_stimuli=2):
    return {
        "weights": np.zeros(num_stimuli),
        "update_fn": no_op_update,
        "predict_fn": zero_model_predict
    }

def zero_model_predict(model, stimulus):
    return np.dot(model["weights"], stimulus)

def no_op_update(model, stimulus, reward, reward_prediction):
    pass 

def rescolra_wagner_create(num_stimuli=2):
    return {
        "weights": np.zeros(num_stimuli),
        "epsilon": .1,
        "update_fn": rescolra_wagner_update,
        "predict_fn": rescolra_wagner_predict
    }

# Side Effect
def rescolra_wagner_update (model, stimulus, reward, reward_prediction):
    """
    rescolra_wagner_update - Update the weights of 
    the model using the rescorla-wagner learning rule.
    stimulus: single stimulus vector 
    """
    delta = reward - reward_prediction 
    update = model["epsilon"] * delta * stimulus
    model["weights"] += update
    print(f"model-weights: {model['weights']}, update: {update}, delta: {delta}, reward: {reward}, reward-prediction: {reward_prediction}, stimulus: {stimulus}, epsilon: {model['epsilon']}")
    return model
    

def rescolra_wagner_predict(model, stimulus):
    return np.dot(model["weights"], stimulus)