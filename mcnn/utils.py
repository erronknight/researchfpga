
import torch
import mcnn.mcnn_model as mcnn

# Generate and Save Dataset


# load in model
def load_model(model_path):
    model = mcnn(*args, **kwargs)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# save model
def save_model(model, model_path):
    try:
        torch.save(model.state_dict(), model_path)
        return True
    except:
        return False
