from CNN1 import CNNModel
from CNN2 import CNNModelWithSkipConnections
from CNN3 import CNNModel2 
from CNN4 import CNNModelWithSkipConnections2
from Train import load_cifar10, train_model
from Experiment import cosine_similarity_between_weights, extract_layer_weights 


model = CNNModel(input_shape=(32, 32, 3), num_classes=10, learning_rate=0.001) 
model2 = CNNModelWithSkipConnections(input_shape=(32, 32, 3), num_classes=10, learning_rate=0.001)
model3 = CNNModel2(input_shape=(32, 32, 3), num_classes=10, learning_rate=0.001)
model4 = CNNModelWithSkipConnections2(input_shape=(32, 32, 3), num_classes=10, learning_rate=0.001)
(train_images, train_labels), (test_images, test_labels) = load_cifar10()



print(" Training the first model") ## the VGG16 style
trained_model = train_model(model.model, train_images, train_labels, test_images, test_labels, epochs=30, batch_size=64)



print(" Training the second model") ## the Resnet style
trained_model2 = train_model(model2.model, train_images, train_labels, test_images, test_labels, epochs=30, batch_size=64)


print(" Training the third model") ## the VGG16 style
trained_model3 = train_model(model3.model, train_images, train_labels, test_images, test_labels, epochs=30, batch_size=64)



print(" Training the fourth model") ## the Resnet style
trained_model4 = train_model(model4.model, train_images, train_labels, test_images, test_labels, epochs=30, batch_size=64)




print("Cosine similarity CNN1") ##to see if the dense layers have a huge impact in the learning process. 
                                ##If less similar more impact.
trained_model_flatten_weights = extract_layer_weights(trained_model, 'dense1')
target_model_flatten_weights = extract_layer_weights(trained_model, 'dense6')
similarity = cosine_similarity_between_weights(trained_model_flatten_weights, target_model_flatten_weights)
print(f"Cosine Similarity: {similarity}")


print("Cosine similarity CNN2") 
trained_model_flatten_weights = extract_layer_weights(trained_model2, 'dense1')
target_model_flatten_weights = extract_layer_weights(trained_model2, 'dense6')
similarity = cosine_similarity_between_weights(trained_model_flatten_weights, target_model_flatten_weights)
print(f"Cosine Similarity: {similarity}")

print("Cosine similarity CNN3") ##Should show more dissimilarity than the same model with less dense.
trained_model_flatten_weights = extract_layer_weights(trained_model3, 'dense1')
target_model_flatten_weights = extract_layer_weights(trained_model3, 'dense12')
similarity = cosine_similarity_between_weights(trained_model_flatten_weights, target_model_flatten_weights)
print(f"Cosine Similarity: {similarity}")


print("Cosine similarity CNN4") 
trained_model_flatten_weights = extract_layer_weights(trained_model4, 'dense1')
target_model_flatten_weights = extract_layer_weights(trained_model4, 'dense12')
similarity = cosine_similarity_between_weights(trained_model_flatten_weights, target_model_flatten_weights)
print(f"Cosine Similarity: {similarity}")

print("Cosine similarity CNN1-CNN2 1st Dense") ##to see if there is a correlation between the  first dense of the two models.
                                               ## this will give insight on if the skip connections have a huge impact on learning
                                               ## independent of the deepth of the dense part.
trained_model_flatten_weights = extract_layer_weights(trained_model, 'dense1')
target_model_flatten_weights = extract_layer_weights(trained_model2, 'dense1')
similarity = cosine_similarity_between_weights(trained_model_flatten_weights, target_model_flatten_weights)
print(f"Cosine Similarity: {similarity}")


print("Cosine similarity CNN1-CNN2 6th Dense") ##this will give inshight on if the models have learned
                                               ##similarly or not (this will make more sense with the accuracy). 
trained_model_flatten_weights = extract_layer_weights(trained_model, 'dense6')
target_model_flatten_weights = extract_layer_weights(trained_model2, 'dense6')
similarity = cosine_similarity_between_weights(trained_model_flatten_weights, target_model_flatten_weights)
print(f"Cosine Similarity: {similarity}")

print("Cosine similarity CNN3-CNN4 1st Dense") ##This should be similar to the same for models 1-2.
trained_model_flatten_weights = extract_layer_weights(trained_model3, 'dense1')
target_model_flatten_weights = extract_layer_weights(trained_model4, 'dense1')
similarity = cosine_similarity_between_weights(trained_model_flatten_weights, target_model_flatten_weights)
print(f"Cosine Similarity: {similarity}")


print("Cosine similarity CNN3-CNN4 12th Dense")
trained_model_flatten_weights = extract_layer_weights(trained_model3, 'dense12')
target_model_flatten_weights = extract_layer_weights(trained_model4, 'dense12')
similarity = cosine_similarity_between_weights(trained_model_flatten_weights, target_model_flatten_weights)
print(f"Cosine Similarity: {similarity}")

print("Cosine similarity CNN1-CNN3 last Dense") ## If depth in dense matters a lot.
trained_model_flatten_weights = extract_layer_weights(trained_model, 'dense6')
target_model_flatten_weights = extract_layer_weights(trained_model3, 'dense12')
similarity = cosine_similarity_between_weights(trained_model_flatten_weights, target_model_flatten_weights)
print(f"Cosine Similarity: {similarity}")


print("Cosine similarity CNN2-CNN4 last Dense") 
trained_model_flatten_weights = extract_layer_weights(trained_model2, 'dense6')
target_model_flatten_weights = extract_layer_weights(trained_model4, 'dense12')
similarity = cosine_similarity_between_weights(trained_model_flatten_weights, target_model_flatten_weights)
print(f"Cosine Similarity: {similarity}")

