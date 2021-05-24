
import torch
from mcnn import mcnn

# Generate and Save Dataset
# see gen_dataset

# load in model
def load_model(model_path):
    model = mcnn()
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

