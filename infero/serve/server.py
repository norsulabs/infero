import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime


class TextRequest(BaseModel):
    text: str


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(
        os.path.join(model_path, "model.onnx"), sess_options
    )
    return tokenizer, session


api_server = FastAPI()

model_path = sys.argv[1] if len(sys.argv) > 1 else ValueError("Model path not provided")
tokenizer, session = load_model(model_path)


@api_server.post("/inference")
async def inference(request: TextRequest):
    try:
        inputs = tokenizer(
            request.text, padding=True, truncation=True, return_tensors="pt"
        )
        ort_inputs = {
            session.get_inputs()[0].name: inputs["input_ids"].numpy(),
            session.get_inputs()[1].name: inputs["attention_mask"].numpy(),
        }
        ort_outs = session.run(None, ort_inputs)
        prediction = ort_outs[0]
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api_server, host="0.0.0.0", port=8000)
