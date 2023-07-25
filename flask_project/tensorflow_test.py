import tensorflow as tf
import subprocess

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    # Check if GPU is available
    if tf.test.gpu_device_name():
        print('GPU is available.')
    else:
        print("GPU is NOT available.")

    # Create and compile the model
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generate some dummy data
    num_samples = 1000
    x_train = tf.random.normal((num_samples, 784))
    y_train = tf.random.uniform((num_samples,), minval=0, maxval=10, dtype=tf.int32)

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # Print GPU usage summary after training
    if tf.test.gpu_device_name():
        gpu_name = tf.test.gpu_device_name()
        print(f"GPU Usage Summary for {gpu_name}:")
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
            print(result.stdout.decode('utf-8'))
        except FileNotFoundError:
            print("nvidia-smi not found. Unable to monitor GPU usage.")
