import concurrent.futures
import tensorflow as tf
import numpy as np
from fgsm_functions import generate_fgsm


# Load the fully connected model
fully_connected_model = tf.keras.models.load_model("C:\\Users\\josep\\OneDrive\\Desktop\\ML2_Project\\AdversarialRobustnessML\\Models\\MNIST\\mnist_model_2LFCN.h5")
# Load the MNIST test dataset
mnist = tf.keras.datasets.mnist
(_, _), (test_images, test_labels) = mnist.load_data()
test_images = test_images / 255.0

# Evaluate the accuracy of the fully connected model with perturbed images
epsilon = 0.15  # You can adjust the epsilon value
with concurrent.futures.ProcessPoolExecutor() as executor:
    perturbed_images = np.array(list(executor.map(lambda image: generate_fgsm(image, fully_connected_model, epsilon), test_images)))

# Evaluate the accuracy of the fully connected model
fully_connected_loss, fully_connected_accuracy = fully_connected_model.evaluate(perturbed_images, test_labels)
print(f"Fully Connected Model - Test Accuracy: {fully_connected_accuracy * 100:.2f}%")