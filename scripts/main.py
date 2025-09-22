from src.utils.reader import load_yaml

if __name__ == '__main__':
    cfg = load_yaml("../configs/default_hparams.yaml")

    # приклад доступу до полів
    backend = cfg["backend"]
    smoke_N = cfg["smoke"]["N"]
    smoke_dt = cfg["smoke"]["dt"]

    print("backend:", backend)
    print("smoke:", {"N": smoke_N, "dt": smoke_dt})

    print(cfg["smoke"])

    pass
