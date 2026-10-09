import tensorflow as tf
import tf2onnx
import numpy as np
import subprocess
import sys

print("=== Konversi Model Keras ke ONNX ===")

# 1. Load model Keras
model = tf.keras.models.load_model('Model/model_lstm.keras')
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# 2. Export ke SavedModel format
saved_model_path = 'Model/saved_model_tmp'
model.export(saved_model_path)
print(f"SavedModel exported to {saved_model_path}")

# 3. Convert SavedModel to ONNX via command line
output_path = 'Model/model_lstm.onnx'
result = subprocess.run([
    sys.executable, '-m', 'tf2onnx.convert',
    '--saved-model', saved_model_path,
    '--output', output_path,
    '--opset', '13'
], capture_output=True, text=True)

print("STDOUT:", result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    # Jika gagal, coba pendekatan manual
    print("\n--- Mencoba pendekatan manual ---")
    
    # Buat concrete function
    @tf.function(input_signature=[tf.TensorSpec([1, 60, 1], dtype=tf.float32)])
    def predict_fn(x):
        return model(x, training=False)
    
    concrete = predict_fn.get_concrete_function()
    
    onnx_model, _ = tf2onnx.convert.from_function(
        concrete,
        input_signature=[tf.TensorSpec([1, 60, 1], dtype=tf.float32, name="input")],
        output_path=output_path
    )
    print("Konversi manual berhasil!")
else:
    print("Konversi berhasil!")

# 4. Validasi
import onnxruntime as ort

session = ort.InferenceSession(output_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

dummy = np.random.rand(1, 60, 1).astype(np.float32)
result_onnx = session.run([output_name], {input_name: dummy})[0]
result_tf = model.predict(dummy, verbose=0)

print(f"\nValidasi:")
print(f"TF output:   {result_tf[0][0]:.6f}")
print(f"ONNX output: {result_onnx[0][0]:.6f}")
print(f"Diff:        {abs(result_tf[0][0] - result_onnx[0][0]):.8f}")
print(f"\nFile ONNX: {output_path}")
