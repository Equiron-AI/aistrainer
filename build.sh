#!/bin/bash

python -m build
twine upload --skip-existing dist/*
