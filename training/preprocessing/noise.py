import numpy as np

def _random_distribution(percentage, size, space=[1]):
    nb_total = size[0] * size[1]
    nb_el = int(nb_total * percentage / 100)
    arr = np.zeros(nb_total)
    for i in range(len(space)):
        start_index = i * nb_el
        arr[start_index:start_index + nb_el] = space[i]

    np.random.shuffle(arr)
    return arr.reshape(size)

def salt_noise(img, percentage):
    rand = _random_distribution(percentage, size=img.shape)
    return np.ma.masked_array(img, rand.reshape(img.shape)).filled(np.max(img))

def pepper_noise(img, percentage):
    rand = _random_distribution(percentage, size=img.shape)
    return np.ma.masked_array(img, rand.reshape(img.shape)).filled(np.min(img))

def salt_pepper_noise(img, percentage):
    rand = _random_distribution(percentage, img.shape, [1, 2])
    arr = np.ma.masked_array(img, (rand == 1).reshape(img.shape)).filled(np.min(img))
    return np.ma.masked_array(arr, (rand == 2).reshape(img.shape)).filled(np.max(img))
