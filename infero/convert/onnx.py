import os
from infero.utils import sanitize_model_name, print_success, print_neutral
from transformers import AutoModelForSequenceClassification
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
import warnings
import logging

logging.getLogger("root").setLevel(
    logging.ERROR
)  ## temporary fix for warning message from onnxruntime


def convert_to_onnx(model_name):

    output_path = f"infero/data/models/{sanitize_model_name(model_name)}/model.onnx"

    if os.path.exists(output_path):
        print_success(f"ONNX model for {model_name} already exists")
        return

    print_neutral(f"Creating ONNX model for {model_name}")

    model = AutoModelForSequenceClassification.from_pretrained(
        f"infero/data/models/{sanitize_model_name(model_name)}/"
    )

    with torch.inference_mode():
        inputs = {
            "input_ids": torch.ones(1, 512, dtype=torch.int64),
            "attention_mask": torch.ones(1, 512, dtype=torch.int64),
        }
        symbolic_names = {0: "batch_size", 1: "max_seq_len"}
        torch.onnx.export(
            model,
            (
                inputs["input_ids"],
                inputs["attention_mask"],
            ),
            output_path,
            opset_version=14,
            input_names=[
                "input_ids",
                "attention_mask",
            ],
            output_names=["output"],
            dynamic_axes={
                "input_ids": symbolic_names,
                "attention_mask": symbolic_names,
            },
        )

        print_success(f"ONNX model for {model_name} created")


def convert_to_onnx_q8(model_name):

    onnx_model_path = f"infero/data/models/{sanitize_model_name(model_name)}/model.onnx"
    quantized_model_path = (
        f"infero/data/models/{sanitize_model_name(model_name)}/model_quantized.onnx"
    )

    if os.path.exists(quantized_model_path):
        print_success(f"Quantized model for {model_name} already exists")
        return

    print_neutral(f"Creating quantized model for {model_name}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Please consider to run pre-processing before quantization",
        )
        quantize_dynamic(
            onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8
        )
    print_success(f"Quantized model for {model_name} created")
