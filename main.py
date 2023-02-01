import os
from fastapi import FastAPI
from pydantic import BaseModel
import replicate
app = FastAPI()

os.environ.get("REPLICATE_API_TOKEN")

class Item(BaseModel):
    image: str

@app.post("/", summary="Generate a caption for an image", tags=["Generate Caption"])
def generate(item: Item):
    model = replicate.models.get("salesforce/blip")
    version = model.versions.get("2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746")
    inputs = {
        'image': item.image,
        'task': "image_captioning",
    }
    output = version.predict(**inputs)
    if output.startswith("Caption:"):
        output = output[8:]
        output = output.strip()

    return output
