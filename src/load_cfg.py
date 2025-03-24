def load_config(config_path):
    with open(config_path) as conf_file:
        cfg = yaml.safe_load(conf_file)
    return cfg