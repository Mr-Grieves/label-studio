import argparse
import json
import logging
import logging.config
import os

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"}
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "stream": "ext://sys.stdout",
                "formatter": "standard",
            }
        },
        "root": {"level": os.getenv("LOG_LEVEL", "INFO"), "handlers": ["console"], "propagate": True},
    }
)

from label_studio_ml.api import init_app

# Imported as the `tsam` package, from the repo root (`python -m tsam.main`), like model.py's own imports.
from tsam.config import load_config
from tsam.model import NewModel

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# label-studio-ml reads LABEL_STUDIO_* (to fetch task images) and mlflow reads MLFLOW_TRACKING_* (to load the model)
# straight from the environment. The VS Code debug config decrypts them into .env before each launch
# (.vscode/tasks.json); from a shell, `eval "$(make -s -C ../isonoai load_secrets)"` first.
_REQUIRED_ENV = (
    "LABEL_STUDIO_URL",
    "LABEL_STUDIO_API_KEY",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    # The handler builds its S3 client at startup, to read the prescan images and blob JSONs.
    "AWS_STORAGE_BUCKET_NAME",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ENDPOINT_URL",
    "AWS_REGION_NAME",
)


def check_env():
    missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise SystemExit(
            f"Missing environment variables: {', '.join(missing)}. "
            "The VS Code debug config loads them into .env first (.vscode/tasks.json); from a shell, run "
            '`eval "$(make -s -C ../isonoai load_secrets)"` first.'
        )


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return dict()
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label studio")
    parser.add_argument("-p", "--port", dest="port", type=int, default=9090, help="Server port")
    parser.add_argument("--host", dest="host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level",
    )
    parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="Switch debug mode")
    parser.add_argument(
        "--check", dest="check", action="store_true", help="Validate model instance before launching server"
    )
    parser.add_argument(
        "--kwargs",
        "--with",
        dest="kwargs",
        metavar="KEY=VAL",
        nargs="+",
        type=lambda kv: kv.split("="),
        help="Additional LabelStudioMLBase model initialization kwargs",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("CONFIG"),
        help="The online-training config, e.g. tsam/configs/foot-ankle.yaml (default: $CONFIG)",
    )
    parser.add_argument(
        "--basic-auth-user", default=os.environ.get("ML_SERVER_BASIC_AUTH_USER", None), help="Basic auth user"
    )
    parser.add_argument(
        "--basic-auth-pass", default=os.environ.get("ML_SERVER_BASIC_AUTH_PASS", None), help="Basic auth pass"
    )

    args = parser.parse_args()
    if not args.config:
        parser.error("--config (or $CONFIG) is required")
    check_env()

    # setup logging level
    if args.log_level:
        logging.root.setLevel(args.log_level)

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def parse_kwargs():
        param = dict()
        for k, v in args.kwargs:
            if v.isdigit():
                param[k] = int(v)
            elif v == "True" or v == "true":
                param[k] = True
            elif v == "False" or v == "false":
                param[k] = False
            elif isfloat(v):
                param[k] = float(v)
            else:
                param[k] = v
        return param

    kwargs = get_kwargs_from_config()

    if args.kwargs:
        kwargs.update(parse_kwargs())

    # Load the model once, up front, so the first request doesn't pay for it and a bad run ID fails at startup.
    NewModel.configure(load_config(args.config))

    if args.check:
        print('Check "' + NewModel.__name__ + '" instance creation..')
        model = NewModel(**kwargs)

    app = init_app(model_class=NewModel, basic_auth_user=args.basic_auth_user, basic_auth_pass=args.basic_auth_pass)

    # No reloader: it re-executes this script in a child process, loading the model twice and escaping the debugger.
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)

else:
    # for uWSGI use
    check_env()
    NewModel.configure(load_config(os.environ["CONFIG"]))
    app = init_app(model_class=NewModel)
