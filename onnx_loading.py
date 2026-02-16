import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("model.onnx")

# Check input name
input_name = session.get_inputs()[0].name
print("Input name:", input_name)

# Check output name
output_name = session.get_outputs()[0].name
print("Output name:", output_name)

# Create dummy input
dummy_input = np.random.randn(1, 24, 250).astype(np.float32)

# Run inference
output = session.run([output_name], {input_name: dummy_input})

print("Output shape:", output[0].shape)
