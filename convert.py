import tensorflow as tf
import tf2onnx

# 1. Load model Keras
model = tf.keras.models.load_model('Model/model_lstm.keras')

# 2. Bungkus model ke dalam tf.function
@tf.function
def model_func(x):
    return model(x)

# 3. Buat concrete function berdasarkan bentuk input model
input_shape = model.input_shape
concrete_func = model_func.get_concrete_function(
    tf.TensorSpec(input_shape, dtype=tf.float32)
)

# 4. Konversi menggunakan from_function (terhindar dari bug Keras)
onnx_model, _ = tf2onnx.convert.from_function(
    concrete_func,
    input_signature=[tf.TensorSpec(input_shape, dtype=tf.float32, name="input")]
)

# 5. Simpan file ONNX
with open('Model/model_lstm.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("Konversi berhasil! File Model/model_lstm.onnx sudah dibuat.")