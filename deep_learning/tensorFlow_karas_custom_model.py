
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras import applications
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from matplotlib import pyplot as plt
from os.path import join


### Provide the training and validation image folder directories created by 'custom_model_img_moving.py' script
# training_folder_dir = r'C:\Users\Koshin\Pictures\car_accident_imgs\BumperDamage\train'
# validation_folder_dir = r'C:\Users\Koshin\Pictures\car_accident_imgs\BumperDamage\val'
training_folder_dir = r"C:\Users\Koshin\PycharmProjects\mask_rcnn_car\mask_rcnn_damage_detection\resized_cropped_car_images_224\BumperDamage\train"
validation_folder_dir = r"C:\Users\Koshin\PycharmProjects\mask_rcnn_car\mask_rcnn_damage_detection\resized_cropped_car_images_224\BumperDamage\val"

### Provide a name for the model that is going to be created
new_model_name = "bumper_damage_front_rear_022719_224pix_00001.h5"

### Provide the path and name of the new model you want to save
new_model_save_dir = r'C:\Users\Koshin\PycharmProjects\logs\bumper_damage20190203custom'

new_model_path = join(new_model_save_dir, new_model_name)


# The functions runs the transfer_learning
def transfer_learning_run():

    # provide the number of classes to classify
    # For instance, glass damage is yes or no; thus, it is 2
    # For Location, it will be 'Urban', 'Suburban', and 'Rural', so it is 3
    # This decides the last layer's number of outputs
    num_classes = 2

    # Creating a model by using Sequential and stack layers on top of it
    my_new_model = Sequential()
    my_new_model.add(applications.InceptionResNetV2(include_top=False, pooling='avg', weights='imagenet'))
    my_new_model.add(Dense(num_classes, activation='softmax'))

    # Not to train first layer (vgg16) model. It is already trained -- freezing vgg16 except for the last layer
    # my_new_model.layers[0].trainable = False
    #
    # print(my_new_model.layers)


    #  to see the model
    my_new_model.summary()

    ### initialize an optimizer -- I chose Adam
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    ### Compile Model
    my_new_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


    ### Fit Model
    # generates images here using ImageDataGenerator
    image_size = 224
    #image_size = 1024
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        horizontal_flip=True,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2)

    # train generator that decides how to pull images from train directory
    train_generator = data_generator.flow_from_directory(
            directory=training_folder_dir,
            target_size=(image_size, image_size),
            batch_size=30,
            class_mode='categorical',
            shuffle=True)

    # For validation, images are not augmented
    data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

    # validation generator
    validation_generator = data_generator_no_aug.flow_from_directory(
            directory=validation_folder_dir,
            target_size=(image_size, image_size),
            batch_size=20,
            class_mode='categorical')

    # train the model
    hs_obj = my_new_model.fit_generator(
            train_generator,
            steps_per_epoch=5,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=2)

    # to indicate the end-result -- training and validation accuracy and cost
    print(hs_obj.history)

    # plot cost and accuracy
    plot_loss(hs_obj.history['loss'], hs_obj.history['val_loss'])
    plot_acc(hs_obj.history['acc'], hs_obj.history['val_acc'])

    # to indicate class indices
    print(train_generator.class_indices)

    # saves the trained model to a designated path
    my_new_model.save(filepath=new_model_path)
    del my_new_model


# To plot both training and testing cost/loss ratio
def plot_loss(train_loss, valid_loss):
    fig, ax = plt.subplots()
    fig_size = [12, 9]
    plt.rcParams["figure.figsize"] = fig_size
    plt.plot(train_loss, label='Training Cost')
    plt.plot(valid_loss, label='Validation Cost')
    plt.title('Cost over time during training')
    legend = ax.legend(loc='upper right')
    plt.show()


# To plot both training and validation accuracy
def plot_acc(train_acc, valid_acc):
    fig, ax = plt.subplots()
    fig_size = [12, 9]
    plt.rcParams["figure.figsize"] = fig_size
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(valid_acc, label='Validation Accuracy')
    plt.title('Accuracy over time during training')
    legend = ax.legend(loc='upper right')
    plt.show()


def main():
    transfer_learning_run()


if __name__ == '__main__':
    main()