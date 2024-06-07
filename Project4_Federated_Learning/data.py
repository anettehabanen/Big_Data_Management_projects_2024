import torch
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

def load_pyp_data(data_dir):
    trainx_path = os.path.join(data_dir, 'trainx.pyp')
    trainy_path = os.path.join(data_dir, 'trainy.pyp')
    testx_path = os.path.join(data_dir, 'testx.pyp')
    testy_path = os.path.join(data_dir, 'testy.pyp')

    x_train = torch.load(trainx_path)
    y_train = torch.load(trainy_path)
    x_test = torch.load(testx_path)
    y_test = torch.load(testy_path)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test
    }

def save_combined_data(client_id, data, output_dir='data/clients'):
    subdir = os.path.join(output_dir, f'{client_id}')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    
    save_path = os.path.join(subdir, 'combined.pt')
    torch.save(data, save_path)
    print(f'Data for client{client_id} saved to {save_path}')

def process_all_clients(num_clients, data_dir='data', output_dir='data/clients'):
    for client_id in range(num_clients):
        data = load_pyp_data(client_id)
        save_combined_data(client_id, data, output_dir)

def load_data(data_path, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/clients/1/combined.pt")

    data = torch.load(data_path)

    if is_train:
        X = data["x_train"]
        y = data["y_train"]
    else:
        X = data["x_test"]
        y = data["y_test"]

    # Normalize
    X = X / 255

    return X, y


# Example usage for 5 clients
if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        process_all_clients(5)
