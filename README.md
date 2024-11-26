# Infero

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/infero)
![PyPI - Version](https://img.shields.io/pypi/v/infero)
![PyPI - Downloads](https://img.shields.io/pypi/dw/infero)
[![CI](https://github.com/norsulabs/infero/actions/workflows/ci.yaml/badge.svg)](https://github.com/norsulabs/infero/actions/workflows/ci.yaml)


## Overview

Infero allows you to easily download, convert, and host your models using the ONNX runtime. It provides a simple CLI to run and maintain the models.

## Features

- Automatic downloads.
- Automatic ONNX conversions.
- Automatic server setup.
- 8-bit quantization support.

## Installation

To install Infero, run the following command:

```bash
pip install infero
```

## Usage

Here is a simple example of how to use Infero:

```bash
infero run [hf_model_name]
```

To run a model with 8-bit quantization:

```bash
infero run [hf_model_name] --quantize
```

To list all available models:

```bash
infero list
```

To remove a model:

```bash
infero remove [hf_model_name]
```

Infero is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or feedback, please contact us at support@norsulabs.com.
