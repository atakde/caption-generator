import os
from fastapi import FastAPI, Request
import replicate
app = FastAPI()

os.environ.get("REPLICATE_API_TOKEN")

@app.get("/")
def read_root(request: Request):
    model = replicate.models.get("salesforce/blip")
    version = model.versions.get("2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746")
    type = request.query_params.get("type") or "url" # todo: add support for base64
    image =  request.query_params.get("image");

    if image is None:
        return {"error": "imageUrl is required"}

    inputs = {
        # Input image
        'image': image,
        # Choose a task.
        'task': "image_captioning",
    }

    output = version.predict(**inputs)
    return output
