from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,BatchNormalization, Dropout

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'Train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'Val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'Test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)



# Compute class weights based on training data
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))


model = Sequential([
    Conv2D(32, (2, 2), activation='relu',padding='same', input_shape=(224, 224, 3)),
    Conv2D(32, (2, 2), activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3),padding='same', activation='relu'),
    Conv2D(64, (3, 3),padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((3, 3)),
    Conv2D(128, (3, 3),padding='same', activation='relu'),
    Conv2D(128, (3, 3),padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((3, 3)),
    Conv2D(256, (5, 5),padding='same', activation='relu'),
    Conv2D(256, (5, 5),padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((5, 5)),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator,epochs=55,validation_data=validation_generator, class_weight=class_weights)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

model.save('lung_cancer_detector.keras')