# https://docs.python.org/3/library/argparse.html
# https://www.youtube.com/watch?v=FbEJN8FsJ9U

from helperUtils import load_model, predict

import numpy as np
import json as js 
import argparse as arg

parser = arg.ArgumentParser()
parser.add_argument('img_path', type=str)
parser.add_argument('model_path', type=str)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--category_names', type=str, required=True)


args = parser.parse_args()

model = load_model(args.model_path)
prob, classes = predict(image_path=args.img_path, model=model, top_k=args.top_k)


with open(args.category_names, 'r') as f:
    class_names = js.load(f)
    classes = {int(i): class_names[str(i)] for i in classes}
    
print('Predictions:', classes)
print('Probability:', prob)
