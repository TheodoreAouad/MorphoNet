import matplotlib.pyplot as plt
import pickle

def get_true_path(path):
    if "file://" in path:
        return path[7:]

    return path

def get_keys(path):
    path = get_true_path(path)
    with open(path, "rb") as f:
        out = pickle.load(f)
        print(out.keys())

class SMorph:
    def plot(data):
        plt.imshow(data['filter'].squeeze(), cmap="plasma")
        plt.title(f"alpha: {data['alpha'].squeeze()}")
        plt.colorbar()
        plt.show()

PLOTABLE_CLASSES = {
    'SMorph': SMorph
}

def plot_sel(path):
    path = get_true_path(path)
    with open(path, "rb") as f:
        saved_data = pickle.load(f)
        for (layer_index, class_name, data) in saved_data['layers_weights']:
            if class_name in PLOTABLE_CLASSES:
                PLOTABLE_CLASSES[class_name].plot(data)
