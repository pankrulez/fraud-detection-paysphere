from src.modeling.train import train_pipeline


def main():
    train_pipeline("config/config.yaml")


if __name__ == "__main__":
    main()