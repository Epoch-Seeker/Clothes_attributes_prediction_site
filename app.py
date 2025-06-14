from fastapi import FastAPI
import keras
import os
import gdown
import numpy as np
import pandas as pd
from typing import List, Union
from fastapi import HTTPException
from fastapi import UploadFile, File , Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import io
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input

# Define the FastAPI app
app = FastAPI()
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Global model variables
tshirt = None
kurti = None
saree = None
top = None
women_tshirts = None

def download_model_from_drive(file_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {dest_path}...")
        gdown.download(url, dest_path, quiet=False)

def build_shared_base():
    input_shape = (128, 128, 3)
    input_layer = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    return input_layer, x

def build_tshirt_model():
    input_layer, x = build_shared_base()
    tshirt_output_attr1 = Dense(5, activation='softmax', name='attr1_output')(x)
    tshirt_output_attr2 = Dense(2, activation='softmax', name='attr2_output')(x)
    tshirt_output_attr3 = Dense(2, activation='softmax', name='attr3_output')(x)
    tshirt_output_attr4 = Dense(3, activation='softmax', name='attr4_output')(x)
    tshirt_output_attr5 = Dense(2, activation='softmax', name='attr5_output')(x)
    return Model(inputs=input_layer, outputs=[tshirt_output_attr1, tshirt_output_attr2, tshirt_output_attr3, tshirt_output_attr4, tshirt_output_attr5])

def build_kurti_model():
    input_layer, x = build_shared_base()
    kurti_output_attr1 = Dense(13, activation='softmax', name='attr1_output')(x)
    kurti_output_attr2 = Dense(2, activation='softmax', name='attr2_output')(x)
    kurti_output_attr3 = Dense(2, activation='softmax', name='attr3_output')(x)
    kurti_output_attr4 = Dense(2, activation='softmax', name='attr4_output')(x)
    kurti_output_attr5 = Dense(2, activation='softmax', name='attr5_output')(x)
    kurti_output_attr6 = Dense(2, activation='softmax', name='attr6_output')(x)
    kurti_output_attr7 = Dense(2, activation='softmax', name='attr7_output')(x)
    kurti_output_attr8 = Dense(3, activation='softmax', name='attr8_output')(x)
    kurti_output_attr9 = Dense(2, activation='softmax', name='attr9_output')(x)
    return Model(inputs=input_layer, outputs=[kurti_output_attr1, kurti_output_attr2, kurti_output_attr3, kurti_output_attr4, kurti_output_attr5, kurti_output_attr6, kurti_output_attr7, kurti_output_attr8, kurti_output_attr9])

def build_saree_model():
    input_layer, x = build_shared_base()
    saree_output_attr1 = Dense(4, activation='softmax', name='attr1_output')(x)
    saree_output_attr2 = Dense(6, activation='softmax', name='attr2_output')(x)
    saree_output_attr3 = Dense(3, activation='softmax', name='attr3_output')(x)
    saree_output_attr4 = Dense(8, activation='softmax', name='attr4_output')(x)
    saree_output_attr5 = Dense(4, activation='softmax', name='attr5_output')(x)
    saree_output_attr6 = Dense(3, activation='softmax', name='attr6_output')(x)
    saree_output_attr7 = Dense(5, activation='softmax', name='attr7_output')(x)
    saree_output_attr8 = Dense(5, activation='softmax', name='attr8_output')(x)
    saree_output_attr9 = Dense(9, activation='softmax', name='attr9_output')(x)
    saree_output_attr10 = Dense(2, activation='softmax', name='attr10_output')(x)
    return Model(inputs=input_layer, outputs=[saree_output_attr1, saree_output_attr2, saree_output_attr3, saree_output_attr4, saree_output_attr5, saree_output_attr6, saree_output_attr7, saree_output_attr8, saree_output_attr9, saree_output_attr10])

def build_top_model():
    input_layer, x = build_shared_base()
    top_output_attr1 = Dense(12, activation='softmax', name='attr1_output')(x)
    top_output_attr2 = Dense(4, activation='softmax', name='attr2_output')(x)
    top_output_attr3 = Dense(2, activation='softmax', name='attr3_output')(x)
    top_output_attr4 = Dense(7, activation='softmax', name='attr4_output')(x)
    top_output_attr5 = Dense(2, activation='softmax', name='attr5_output')(x)
    top_output_attr6 = Dense(3, activation='softmax', name='attr6_output')(x)
    top_output_attr7 = Dense(6, activation='softmax', name='attr7_output')(x)
    top_output_attr8 = Dense(4, activation='softmax', name='attr8_output')(x)
    top_output_attr9 = Dense(4, activation='softmax', name='attr9_output')(x)
    top_output_attr10 = Dense(6, activation='softmax', name='attr10_output')(x)
    return Model(inputs=input_layer, outputs=[top_output_attr1, top_output_attr2, top_output_attr3, top_output_attr4, top_output_attr5, top_output_attr6, top_output_attr7, top_output_attr8, top_output_attr9, top_output_attr10])

def build_women_tshirts_model():
    input_layer, x = build_shared_base()
    women_tshirts_output_attr1 = Dense(7, activation='softmax', name='attr1_output')(x)
    women_tshirts_output_attr2 = Dense(3, activation='softmax', name='attr2_output')(x)
    women_tshirts_output_attr3 = Dense(3, activation='softmax', name='attr3_output')(x)
    women_tshirts_output_attr4 = Dense(3, activation='softmax', name='attr4_output')(x)
    women_tshirts_output_attr5 = Dense(6, activation='softmax', name='attr5_output')(x)
    women_tshirts_output_attr6 = Dense(3, activation='softmax', name='attr6_output')(x)
    women_tshirts_output_attr7 = Dense(2, activation='softmax', name='attr7_output')(x)
    women_tshirts_output_attr8 = Dense(2, activation='softmax', name='attr8_output')(x)
    return Model(inputs=input_layer, outputs=[women_tshirts_output_attr1 , women_tshirts_output_attr2 , women_tshirts_output_attr3 , women_tshirts_output_attr4, women_tshirts_output_attr5, women_tshirts_output_attr6, women_tshirts_output_attr7 , women_tshirts_output_attr8])

@app.on_event("startup")
def load_models():
    global tshirt, kurti, saree, top, women_tshirts

    download_model_from_drive('1OZBQccslqN5HdviewYFD4WxIovdHxkyP', f'{MODEL_DIR}/tshirt_model.keras')
    download_model_from_drive('175PtqO7e8J_Uc_RYPG2kbmTKZWwTCfdw', f'{MODEL_DIR}/kurti_weights.keras')
    download_model_from_drive('1GKq45ljHniW2IPXN7lcTXSagc_A6PlZc', f'{MODEL_DIR}/saree_weights.keras')
    download_model_from_drive('1km6cDHpLiCwDFkqjXgWsW49jRPgpKd8V', f'{MODEL_DIR}/women_top_weights.keras')
    download_model_from_drive('12wIMgxpIXDThJxquNv_b79hnHn0ocSgp', f'{MODEL_DIR}/women_tshirts_weights.keras')

    tshirt = build_tshirt_model()
    kurti = build_kurti_model()
    saree = build_saree_model()
    top = build_top_model()
    women_tshirts = build_women_tshirts_model() 
    tshirt.load_weights(f'{MODEL_DIR}/tshirt_model.keras')
    print("T-shirt model weights loaded.")
    kurti.load_weights(f'{MODEL_DIR}/kurti_weights.keras')
    print("Kurti model weights loaded.")
    saree.load_weights(f'{MODEL_DIR}/saree_weights.keras')
    print("Saree model weights loaded.")
    top.load_weights(f'{MODEL_DIR}/top_weights.keras')
    print("top model weights loaded.")
    women_tshirts.load_weights(f'{MODEL_DIR}/women_tshirts_weights.keras') 
    print("women_tshirts model weights loaded.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(category: str = Form(...),file: UploadFile = File(...) ):

    contents = await file.read()

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    image_data = img.resize((128, 128))

    img = np.array(image_data).astype('float32') / 255.0

    img_array = np.expand_dims(img, axis=0)

    if category.lower() == "tshirt":
        predictions = tshirt.predict(img_array)
        attr1, attr2, attr3, attr4, attr5 = predictions
        attr1_classes =['default', 'multicolor', 'black', 'white', 'dummy_value'] 
        attr2_classes = ['round', 'polo'] 
        attr3_classes = ['printed', 'solid']  
        attr4_classes = ['default', 'solid', 'typography']    
        attr5_classes = ['short sleeves', 'long sleeves'] 
        attr1_pred = attr1_classes[np.argmax(attr1)]
        attr2_pred = attr2_classes[np.argmax(attr2)]
        attr3_pred = attr3_classes[np.argmax(attr3)]
        attr4_pred = attr4_classes[np.argmax(attr4)]
        attr5_pred = attr5_classes[np.argmax(attr5)]
        return {
            "Color": attr1_pred,
            "Neck": attr2_pred,
            "Print": attr3_pred,
            "Design": attr4_pred,
            "Sleeve": attr5_pred
        }
    
    elif category.lower() == "kurti":
        predictions = kurti.predict(img_array)
        (attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9) = predictions
        attr1_classes = ['black', 'red', 'navy blue', 'maroon', 'green', 'pink', 'blue', 'purple', 'grey', 'yellow', 'white', 'multicolor', 'orange']
        attr2_classes = ['straight', 'a-line']
        attr3_classes = ['knee length', 'calf length']
        attr4_classes = ['daily', 'party']
        attr5_classes = ['net', 'default']
        attr6_classes = ['solid', 'default']
        attr7_classes = ['solid', 'default']
        attr8_classes = ['three-quarter sleeves', 'short sleeves', 'sleeveless']
        attr9_classes = ['regular', 'sleeveless']
        attr1_pred = attr1_classes[np.argmax(attr1)]
        attr2_pred = attr2_classes[np.argmax(attr2)]
        attr3_pred = attr3_classes[np.argmax(attr3)]
        attr4_pred = attr4_classes[np.argmax(attr4)]
        attr5_pred = attr5_classes[np.argmax(attr5)]
        attr6_pred = attr6_classes[np.argmax(attr6)]
        attr7_pred = attr7_classes[np.argmax(attr7)]
        attr8_pred = attr8_classes[np.argmax(attr8)]
        attr9_pred = attr9_classes[np.argmax(attr9)]
        return {
            "Color": attr1_pred,
            "Dress Shape": attr2_pred,
            "Dress length": attr3_pred,
            "Occasion": attr4_pred,
            "Fabric": attr5_pred,
            "Print 1": attr6_pred,
            "Print 2": attr7_pred,
            "Sleeve": attr8_pred,
            "Fit": attr9_pred
        }
    
    elif category.lower() == "saree":
        predictions = saree.predict(img_array)
        (attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10) = predictions
        attr1_classes = ['same as saree', 'solid', 'same as border', 'default']
        attr2_classes = ['woven design', 'zari', 'no border', 'solid', 'default', 'temple border']
        attr3_classes = ['small border', 'big border', 'no border']
        attr4_classes = ['multicolor', 'cream', 'white', 'default', 'navy blue', 'yellow', 'green', 'pink']
        attr5_classes = ['party', 'traditional', 'daily', 'wedding']
        attr6_classes = ['jacquard', 'default', 'tassels and latkans']
        attr7_classes = ['woven design', 'dummy_value', 'same as saree', 'default', 'zari woven']
        attr8_classes = ['zari woven', 'woven design', 'default', 'solid', 'printed']
        attr9_classes = ['applique', 'elephant', 'floral', 'ethnic motif', 'default', 'peacock', 'solid', 'checked', 'botanical']
        attr10_classes = ['no', 'yes']
        attr1_pred = attr1_classes[np.argmax(attr1)]
        attr2_pred = attr2_classes[np.argmax(attr2)]
        attr3_pred = attr3_classes[np.argmax(attr3)]
        attr4_pred = attr4_classes[np.argmax(attr4)]
        attr5_pred = attr5_classes[np.argmax(attr5)]
        attr6_pred = attr6_classes[np.argmax(attr6)]
        attr7_pred = attr7_classes[np.argmax(attr7)]
        attr8_pred = attr8_classes[np.argmax(attr8)]
        attr9_pred = attr9_classes[np.argmax(attr9)]
        attr10_pred = attr10_classes[np.argmax(attr10)]
        return {
            "Blouse Pattern": attr1_pred,
            "Border": attr2_pred,
            "Border Size": attr3_pred,
            "Color": attr4_pred,
            "Occasion": attr5_pred,
            "Design Detail": attr6_pred,
            "Pallu Design": attr7_pred,
            "Saree Body Design": attr8_pred,
            "Print Pattern": attr9_pred,
            "Has Blouse": attr10_pred
        }
    
    elif category.lower() == "Women Top":
        predictions = top.predict(img_array)
        (attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10) = predictions
        attr1_classes = ['black', 'navy blue', 'red', 'default', 'maroon', 'white', 'green', 'blue', 'pink', 'yellow', 'peach', 'multicolor']
        attr2_classes = ['regular', 'fitted', 'boxy', 'default']
        attr3_classes = ['regular', 'crop']
        attr4_classes = ['round neck', 'high', 'stylised', 'sweetheart neck', 'v-neck', 'square neck', 'default']
        attr5_classes = ['casual', 'party']
        attr6_classes = ['solid', 'default', 'printed']
        attr7_classes = ['solid', 'typography', 'graphic', 'default', 'quirky', 'floral']
        attr8_classes = ['short sleeves', 'sleeveless', 'three-quarter sleeves', 'long sleeves']
        attr9_classes = ['regular sleeves', 'default', 'sleeveless', 'puff sleeves']
        attr10_classes = ['knitted', 'default', 'ruffles', 'waist tie-ups', 'tie-ups', 'applique']
        attr1_pred = attr1_classes[np.argmax(attr1)]
        attr2_pred = attr2_classes[np.argmax(attr2)]
        attr3_pred = attr3_classes[np.argmax(attr3)]
        attr4_pred = attr4_classes[np.argmax(attr4)]
        attr5_pred = attr5_classes[np.argmax(attr5)]
        attr6_pred = attr6_classes[np.argmax(attr6)]
        attr7_pred = attr7_classes[np.argmax(attr7)]
        attr8_pred = attr8_classes[np.argmax(attr8)]
        attr9_pred = attr9_classes[np.argmax(attr9)]
        attr10_pred =  attr10_classes[np.argmax(attr10)]
        return {
            "Color":  attr1_pred,
            "Fit":  attr2_pred,
            "Length":  attr3_pred,
            "Neck":  attr4_pred,
            "Occasion":  attr5_pred,
            "Print":  attr6_pred,
            "Design":  attr7_pred,
            "Sleeve Length":  attr8_pred,
            "Sleeve Style":  attr9_pred,
            "Extra Design":  attr10_pred
        }

    elif category.lower() == "Women T-shirt":
        predictions = women_tshirts.predict(img_array)
        (attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8) = predictions
        attr1_classes = ['multicolor', 'yellow', 'black', 'default', 'pink', 'maroon', 'white']
        attr2_classes = ['loose', 'boxy', 'regular']
        attr3_classes = ['long', 'crop', 'regular']
        attr4_classes = ['default', 'solid', 'printed']
        attr5_classes = ['default', 'quirky', 'solid', 'graphic', 'funky print', 'typography']
        attr6_classes = ['default', 'long sleeves', 'short sleeves']
        attr7_classes = ['regular sleeves','cuffed sleeves']
        attr8_classes = [ 'default', 'applique']
        attr1_pred = attr1_classes[np.argmax(attr1)]
        attr2_pred = attr2_classes[np.argmax(attr2)]
        attr3_pred = attr3_classes[np.argmax(attr3)]
        attr4_pred = attr4_classes[np.argmax(attr4)]
        attr5_pred = attr5_classes[np.argmax(attr5)]
        attr6_pred = attr6_classes[np.argmax(attr6)]
        attr7_pred = attr7_classes[np.argmax(attr7)]
        attr8_pred = attr8_classes[np.argmax(attr8)]
        return {
            "Color": attr1_pred,
            "Fit": attr2_pred,
            "Length": attr3_pred,
            "Print": attr4_pred,
            "Design": attr5_pred,
            "Sleeve Length": attr6_pred,
            "Sleeve Style": attr7_pred,
            "Extra Design": attr8_pred
        }
    
@app.get("/")
def health_check():
    return {"status": "ready"}

@app.get("/")
def serve_home():
    return FileResponse("home.html")
