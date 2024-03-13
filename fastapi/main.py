from fastapi import FastAPI, UploadFile, Form
from PIL import Image
import torch
from io import BytesIO
from typing import List
from model_utils import (
    init_rgb_model,
    init_semseg_model,
    init_rgb_semseg_model,
    inference,
    semseg_inference,
    multi_inference,
    prepare_image,
    target_transform,
    masked_berhu_loss,
)

app = FastAPI()
rgb_model = init_rgb_model()
semseg_model = init_semseg_model()
multi_model = init_rgb_semseg_model()

authorized_tokens = ["my_token"]


@app.get("/")
def root():
    return {"message": "Successful Response"}


@app.get("/rgb_inference/")
async def rgb_inference(rgb: UploadFile):
    contents_rgb = await rgb.read()
    pil_image_rgb = Image.open(BytesIO(contents_rgb)).convert("RGB")

    prediction = inference(pil_image_rgb, rgb_model)

    return {"prediction": prediction.tolist()}


@app.get("/semantic_inference/")
async def semantic_inference(semseg: UploadFile):
    contents_semseg = await semseg.read()
    pil_image_semseg = Image.open(BytesIO(contents_semseg)).convert("L")

    prediction = semseg_inference(pil_image_semseg, semseg_model)

    return {"prediction": prediction.tolist()}


@app.get("/rgb_semantic_inference/")
async def rgb_semantic_inference(rgb: UploadFile, semseg: UploadFile):
    contents_semseg = await semseg.read()
    pil_image_semseg = Image.open(BytesIO(contents_semseg)).convert("L")

    contents_rgb = await rgb.read()
    pil_image_rgb = Image.open(BytesIO(contents_rgb)).convert("RGB")

    prediction = multi_inference(pil_image_rgb, pil_image_semseg, multi_model)

    return {"prediction": prediction.tolist()}


@app.post("/fine_tune_rgb/")
async def fine_tune_rgb(
    rgb_ls: List[UploadFile],
    depth_ls: List[UploadFile],
    token: str = Form(...)
):
    if token not in authorized_tokens:
        return {"message": "Fine tuning is not permitted"}
    if not len(rgb_ls) < 8:
        return {"message": "Too much samples for a small service"}
    if not (len(rgb_ls) == len(depth_ls)):
        return {"message": "All Images should have target"}

    losses = await fine_tune(rgb_ls, depth_ls)

    return {"message": "RGB Model successfully fine-tuned", "losses": losses}


async def fine_tune(rgb_ls, depth_ls):
    global rgb_model

    rgb_model.train()
    losses = []
    optimizer = torch.optim.Adam(rgb_model.parameters(), 3e-5)

    for img, target in zip(rgb_ls, depth_ls):
        contents_rgb = await img.read()
        pil_image_rgb = Image.open(BytesIO(contents_rgb)).convert("RGB")

        contents_depth = await target.read()
        target = Image.open(BytesIO(contents_depth))
        ground_truth = target_transform(target)

        sample_dict = prepare_image(pil_image_rgb)

        optimizer.zero_grad()
        out = rgb_model(sample_dict, return_all_layers=True)
        loss = masked_berhu_loss(preds=out["depth"], target=ground_truth)

        loss.backward()
        losses.append(loss.cpu().detach().item())

        optimizer.step()

    rgb_model.eval()

    return losses


@app.get("/rgb_embeddings/")
async def rgb_embedding(rgb: UploadFile):
    contents_rgb = await rgb.read()
    pil_image_rgb = Image.open(BytesIO(contents_rgb)).convert("RGB")
    sample_dict = prepare_image(pil_image_rgb)

    with torch.no_grad():
        input_tokens, input_info = rgb_model.process_input(sample_dict)
        encoder_tokens = rgb_model.encoder(input_tokens)[0].cpu().numpy()

    return {
        "message": "RGB embeddings successfully extracted",
        "embeddings": encoder_tokens.tolist(),
    }
