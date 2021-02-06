from skimage import io


def _get_instances(dataset_path, filter_predicate=lambda inst: True):
    f = open(f"{dataset_path}/Scene_Instances.txt")
    instances_str = list(
        filter(lambda l: len(l) != 0, map(lambda l: l.strip(), f.read().split("\n")))
    )

    def annotate_instance(l):
        parts = l.split("_")
        return {
            "path": l,
            "scene_number": parts[1],
            "smartphone_code": parts[2],
            "iso_level": int(parts[3]),
            "shutter_speed": int(parts[4]),
            "illuminant": parts[5],
            "illuminant_brightness_code": parts[6],
        }

    return list(filter(filter_predicate, map(annotate_instance, instances_str)))


def _get_noisy_path(dataset_path, inst):
    path = inst["path"]
    dirpath = f"{dataset_path}/Data/{path}"
    return f"{dirpath}/NOISY_SRGB_010.PNG"


def _get_gt_path(dataset_path, inst):
    path = inst["path"]
    dirpath = f"{dataset_path}/Data/{path}"
    return f"{dirpath}/GT_SRGB_010.PNG"


def _load_noisy(instances, dataset_path):
    def load_instance(inst):
        return io.imread(_get_noisy_path(dataset_path, inst))

    noisy = map(load_instance, instances)

    return noisy


def _load_gt(instances, dataset_path):
    def load_instance(inst):
        return io.imread(_get_gt_path(dataset_path, inst))

    gt = map(load_instance, instances)

    return gt


def load_noisy(
    dataset_path, filter_predicate=None,
):
    instances = _get_instances(dataset_path, filter_predicate)

    return _load_noisy(instances, dataset_path)


def load_gt(
    dataset_path, filter_predicate=None,
):
    instances = _get_instances(dataset_path, filter_predicate)

    return _load_gt(instances, dataset_path)


def load_data(
    dataset_path, filter_predicate=None,
):
    instances = _get_instances(dataset_path, filter_predicate)

    instances_y, instances_x = zip(
        _load_gt(instances, dataset_path), _load_noisy(instances, dataset_path),
    )

    return instances_x, instances_y
