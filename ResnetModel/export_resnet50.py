import tensorflow as tf

model = tf.keras.applications.ResNet50(weights="imagenet")

# Give the model a named output
outputs = {'output': model.output}

new_model = tf.keras.Model(inputs=model.input, outputs=outputs)

tf.saved_model.save(new_model, "new_savedmodel")
