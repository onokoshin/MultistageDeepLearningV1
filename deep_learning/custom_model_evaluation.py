from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np
from os.path import join
import pandas as pd
import os
from PIL import Image as im

### provide a name of the model in h5 format here
model_name = 'bumper_damage_front_rear_021019_224pix_vgg19_00001.h5'

### provide the directory path to the model you want to evaluate
model_dir = r"C:\Users\Koshin\PycharmProjects\logs\bumper_damage20190203custom"

model_path = join(model_dir, model_name)

### provie the directory path to where all the testing images are located
img_folder_dir = r"C:\Users\Koshin\PycharmProjects\mask_rcnn_car\mask_rcnn_damage_detection\resized_cropped_car_images_224"

### provide the directory path to testing.csv file
testing_df = r"C:\Users\Koshin\PycharmProjects\deep_learning/testing_car.csv"

### provide one of the classifier names here
classifier = 'BumperDamage'


'''
Global Constants: classifier answer selection options
'''
CONST_YES = 'yes'
CONST_NO = 'no'
CONST_URBAN = 'urban'
CONST_SUBURBAN = 'suburban'
CONST_RURAL = 'rural'
CONST_DAY = 'day'
CONST_NIGHT = 'night'
CONST_CLEAR = 'clear'
CONST_RAINY = 'rainy'
CONST_SNOWY = 'snowy'
CONST_DRY = 'dry'
CONST_WET = 'wet'
CONST_ZERO = 0
CONST_ONE = 1
CONST_TWO = 2
CONST_THREE = 3
CONST_FOUR = 4


# reads data from csv file and convert it to pandas dataFrame
def csv_to_dataFrame(testing_df_path):

    with open(testing_df_path) as file:

        #reads the csv file
        df = pd.read_csv(file)

        df = df_data_cleaning(df)

    return df

def df_data_cleaning(df):

    # drop all NaN values if exists
    df = df.dropna(thresh=2)

    # assigns the first column name to a variable
    fileName = df.columns.values[0]

    # naming validation
    if fileName != 'ImageFilename':
        # rename the first columns since it contains glitchy characters
        df.columns.values[0] = 'ImageFilename'

    classifier_list = list(df.columns.values)[3:16]

    # ensure that every attribute is in lower-case and there is no space after
    for row_index, row in df.iterrows():

        for classifier in classifier_list:

            # lower-casing and data cleaning apply to string input e.g. 'rural', 'yes', and etc
            if type(row[classifier]) == str:
                lower_case_value = row[classifier].lower()
                df.at[row_index, classifier] = lower_case_value

                if row[classifier][0] == 'y' and row[classifier][1] == 'e':
                    df.at[row_index, classifier] = CONST_YES
                elif row[classifier][0] == 'n' and row[classifier][1] == 'o':
                    df.at[row_index, classifier] = CONST_NO
                elif row[classifier][0] == 'u' and row[classifier][1] == 'r':
                    df.at[row_index, classifier] = CONST_URBAN
                elif row[classifier][0] == 's' and row[classifier][1] == 'u':
                    df.at[row_index, classifier] = CONST_SUBURBAN
                elif row[classifier][0] == 'r' and row[classifier][1] == 'u':
                    df.at[row_index, classifier] = CONST_RURAL
                elif row[classifier][0] == 'd' and row[classifier][1] == 'a':
                    df.at[row_index, classifier] = CONST_DAY
                elif row[classifier][0] == 'n' and row[classifier][1] == 'i':
                    df.at[row_index, classifier] = CONST_NIGHT
                elif row[classifier][0] == 'c' and row[classifier][1] == 'l':
                    df.at[row_index, classifier] = CONST_CLEAR
                elif row[classifier][0] == 'r' and row[classifier][1] == 'a':
                    df.at[row_index, classifier] = CONST_RAINY
                elif row[classifier][0] == 's' and row[classifier][1] == 'n':
                    df.at[row_index, classifier] = CONST_SNOWY
                elif row[classifier][0] == 'd' and row[classifier][1] == 'r':
                    df.at[row_index, classifier] = CONST_DRY
                elif row[classifier][0] == 'w' and row[classifier][1] == 'e':
                    df.at[row_index, classifier] = CONST_WET

    return df


def run():

    my_model = load_model(filepath=model_path)

    testing_car_df = csv_to_dataFrame(testing_df_path=testing_df)

    img_name_list = list()

    # instantiate a dictionary and fill up keys
    for row_index, row in testing_car_df.iterrows():
        image_name = row['ImageFilename']
        # img_dict[image_name] = list()
        img_name_list.append(image_name)

    column_list = ['ImageFilename', classifier]

    result_df = pd.DataFrame(columns=column_list)
    df_list = list()

    for img_name in img_name_list:
        test_img_path = join(img_folder_dir, img_name)


        # img_path = 'glass_model_test_sample/no_damage_0.jpg'
        img = image.load_img(path=test_img_path, grayscale=False, target_size=(224, 224))
        x = image.img_to_array(img)
        # print('image_to_array: ', x)
        # print(x.shape)
        x = np.expand_dims(x, axis=0)
        # print('np_expand_dims', x)
        # print(x.shape) # current shape -- (1, 224, 224, 3)
        ## the image is now prepared -- for CNN, input must be a 4-D tensor [batch_size, width, height, channel]
        ## channel -- 1: gray scale, 3: RGB(red, green glue)

        prediction = my_model.predict(x)
        #print(test_img_path)
        #print('NoBumperDamage', '  ', 'YesBumperDamage')
        #print(prediction)
        print(prediction[0][0], prediction[0][1])
        no_val = prediction[0][0]
        yes_val = prediction[0][1]

        if no_val >= yes_val:
            eval = 'no'
        else:
            eval = 'yes'

        one_row_data = [img_name, eval]

        one_row_df = pd.DataFrame(data=[one_row_data], columns=column_list)

        result_df = result_df.append(one_row_df, ignore_index=True)

    csv_output(df=result_df, csv_output_name='bumper_damage_result.csv')


def csv_output(df, csv_output_name):

    print('\n{}.csv file is being created...'.format(csv_output_name))
    while True:
        try:
            # compare the reult_df with testing_car_df
            df.to_csv(csv_output_name, sep=',', index=False)
            print('Successfully created {} file!'.format(csv_output_name))
            break

        except PermissionError:
            print("Encountered Permission Error: please take an appropriate action!")
            print("\nEnter 'yes' to proceed or enter anything else to exit the program")
            answer = input()
            answer = answer.lower()

            if answer[0] != 'y':
                print('Ok, goodbye')
                exit()


def individual_testing():

    test_img_path = r'C:\Users\Koshin\PycharmProjects\mask_rcnn_car\mask_rcnn_damage_detection\resized_cropped_car_images_1024\car-accident-00309.jpeg'
    my_model = load_model(filepath=model_path)

    img = im.open(test_img_path)
    img.show()


    img = image.load_img(path=test_img_path, grayscale=False, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    prediction = my_model.predict(x)
    print('NoBumperDamage', '  ', 'YesBumperDamage')
    # print(prediction)
    print(prediction[0][0], prediction[0][1])


def bulk_testing():

    DIR_PATH = r"C:\Users\Koshin\PycharmProjects\mask_rcnn_car\mask_rcnn_damage_detection\resized_cropped_car_images_1024"
    ls = os.listdir(DIR_PATH)[1:]
    my_model = load_model(filepath=model_path)
    start = 303
    end = 320

    start -= 1


    for img_path in ls[start:end]:
        test_img_path = os.path.join(DIR_PATH, img_path)
        # print(test_img_path)

        #test_img_path = r'C:\Users\Koshin\PycharmProjects\mask_rcnn_car\mask_rcnn_damage_detection\resized_cropped_car_images_1024\car-accident-00111.jpg'
        img = image.load_img(path=test_img_path, grayscale=False, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        prediction = my_model.predict(x)
        #print(img_path)
        #print('NoBumperDamage', '  ', 'YesBumperDamage')
        # print(prediction)
        print(prediction[0][0], prediction[0][1])



def main():
    run()

    # individual_testing()
    #
    # bulk_testing()

if __name__ == '__main__':
    main()