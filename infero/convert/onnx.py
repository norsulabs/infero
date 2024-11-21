import os
from infero.pull.download import download_model
from infero.utils import sanitize_model_name, print_success
from transformers import AutoModelForSequenceClassification
import torch


def convert_to_onnx(model_name):

    output_path = f"infero/data/models/{sanitize_model_name(model_name)}/model.onnx"

    if os.path.exists(output_path):
        print_success(f"ONNX model for {model_name} already exists")
        return
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
