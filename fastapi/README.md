It is a fastapi app for the multimae project, that provides API for working with model that predicts depth from multiple input modalities.

# Usage

One should first have directory /ckpt with checkpoints located there. They are build in the right place if installing from the docker image, described in main README

All the APIs are availible at main.py

### Main Functions

1) Inference for RGB model

GET method at ```/rgb_inference/```
image in RGB format should be provided. The example could be seen in tests.py. For examle, running from python

```python
rgb_path = "your rgb image path"
files = {"rgb": open(rgb_path, "rb")}
example_url = "http://127.0.0.1:8886/rgb_inference/"
response = requests.get(example_url, files=files)
```
The response json will contain "prediction" that is a prediction of a model in a format of python list.

2) Inference for Semantic Mask model

GET method at ```/semantic_inference/```
Semantic Mask image format should be provided. The same logic as in RGB inference

3) Inference for RGB+Semantic Mask model

GET method at ```/rgb_semantic_inference/```
Both RGB image and Semantic mask should be provided. The usage is the same as previous inference methods. For examle, running from python

```python
rgb_path = "your rgb path"
semseg_path = "your semseg path"
files = {"rgb": open(rgb_path, "rb"), "semseg": open(semseg_path, "rb")}
url = "http://127.0.0.1:8886/rgb_semantic_inference/"
response = requests.get(url, files=files)
```

4) Obtain embeddings of MultiMAE without prediction

GET method at ```/rgb_embeddings/```
Expects RGB image to be provided the same way as in RGB inference. Output contains "embeddings" key that contains embedding in type of python list

5) Fine-tune RGB only model (Beta)

POST method at ```/fine_tune_rgb/```
Fine-tunes loaded model with provided images and corresponding targets. Access Token Should be provided as well. For now parameters of fine-tuning could not be changed (will fix it later). For examle, running from python

```python

rgb_path = "your rgb path"
depth_path = "your depth path"
url = "http://127.0.0.1:8886/fine_tune_rgb/"

files = []
# add rgb image
files.append(
    ("rgb_ls",
        (f"rgb_image{i}.png", open(rgb_path, "rb"),
            "image/png"))
)
# add depth image
files.append(
    ("depth_ls",
        (f"depth_image{i}.png", open(depth_path, "rb"),
            "image/png"))
)
# set auth token
data = {"token": "my_token"}

response = requests.post(url, files=files, data=data)
```
The response will contain message and list with losses in "losses" key. If the fine-tuning message is succesful, then current model is already fine-tuned
