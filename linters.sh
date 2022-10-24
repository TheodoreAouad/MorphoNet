python -m black ./src
python -m pylint --rc-file pyproject.toml ./src
python -m mypy --config-file pyproject.toml ./src