import shutil
import pandas as pd
import os
import sklearn.utils as sku

### Provide training_csv and testing_csv file directories
### These two training_csv and testing csv should be the output of process_data.py
training_csv_src = r"/Users/Koshin/PycharmProjects\deep_learning\training_car.csv"
testing_csv_src = r"/Users/Koshin/PycharmProjects/deep_learning/testing_car.csv"

### Provide the directory to where all the images of car-accident are located
# src_folder_path = r"/Users/Koshin/Pictures/car_accident_imgs"
src_folder_path = r"C:\Users\Koshin\PycharmProjects\mask_rcnn_car\mask_rcnn_damage_detection\resized_cropped_car_images_224"



# reads data from csv file and convert it to pandas dataFrame
def csv_to_dataFrame(csv_src):

    with open(csv_src) as file:

        #reads the csv file
        df = pd.read_csv(file)

        #assigns the first column name to a variable
        fileName = df.columns.values[0]

        #naming validation
        if fileName != 'ImageFilename':
            #rename the first columns since it contains glitchy characters
            df.columns.values[0] = 'ImageFilename'

    return df


# takes classifir/feature name and dataFrame from excel file to process data
def process_data(classifier, training_df, testing_df):

    if classifier == 'GlassDamage' or classifier == 'WaterDamage' or classifier == 'BumperDamage' \
        or classifier == 'RolledOver' or classifier == 'SideDoorDamage' or classifier == 'SlidOffRoad' \
        or classifier == 'Bumps&Dings' or classifier == 'EngineDamage':

        boolean_divider(training_df, classifier, is_train=True)
        boolean_divider(testing_df, classifier, is_train=False)

    elif classifier == 'Location':
        location_divider(training_df, classifier, is_train=True)
        location_divider(testing_df, classifier, is_train=False)

    elif classifier == 'Time':
        time_divider(training_df, classifier, is_train=True)
        time_divider(testing_df, classifier, is_train=False)

    elif classifier == 'Weather':
        weather_divider(training_df, classifier, is_train=True)
        weather_divider(testing_df, classifier, is_train=False)

    elif classifier == 'RoadCondition':
        roadCondition_divider(training_df, classifier, is_train=True)
        roadCondition_divider(testing_df, classifier, is_train=False)

    elif classifier == 'Severity':
        severity_divider(training_df, classifier, is_train=True)
        severity_divider(training_df, classifier, is_train=False)



def boolean_divider(df, classifier, is_train):

    # selecte rows that apply to the condition both yes/no
    dfYes = df[df[classifier] == 'yes']
    dfNo = df[df[classifier] == 'no']

    # store selection in variables
    y = 'Yes'
    n = 'No'

    move_file(dfYes, y, classifier, is_train)
    move_file(dfNo, n, classifier, is_train)


def location_divider(df, classifier, is_train):

    # selecte rows that apply to the condition both yes/no
    dfUrban = df[df[classifier] == 'urban']
    dfSuburban = df[df[classifier] == 'suburban']
    dfRural = df[df[classifier] == 'rural']

    # store selection in variables
    u = 'Urban'
    s = 'Suburban'
    r = 'Rural'

    move_file(dfUrban, u, classifier, is_train)
    move_file(dfSuburban, s, classifier, is_train)
    move_file(dfRural, r, classifier, is_train)


def time_divider(df, classifier, is_train):

    # selecte rows that apply to the condition both yes/no
    dfDay = df[df[classifier] == 'day']
    dfNight = df[df[classifier] == 'night']


    # store selection in variables
    d = 'Day'
    n = 'Night'

    move_file(dfDay, d, classifier, is_train)
    move_file(dfNight, n, classifier, is_train)


def weather_divider(df, classifier, is_train):

    # selecte rows that apply to the condition both yes/no
    dfClear = df[df[classifier] == 'clear']
    dfRainy = df[df[classifier] == 'rainy']
    dfSnowy = df[df[classifier] == 'snowy']

    # store selection in variables
    c = 'Clear'
    r = 'Rainy'
    s = 'Snowy'

    move_file(dfClear, c, classifier, is_train)
    move_file(dfRainy, r, classifier, is_train)
    move_file(dfSnowy, s, classifier, is_train)


def roadCondition_divider(df, classifier, is_train):

    # selecte rows that apply to the condition both yes/no
    dfDry = df[df[classifier] == 'dry']
    dfWet = df[df[classifier] == 'wet']
    dfSnowy = df[df[classifier] == 'snowy']

    # store selection in variables
    d = 'Dry'
    w = 'Wet'
    s = 'Snowy'

    move_file(dfDry, d, classifier, is_train)
    move_file(dfWet, w, classifier, is_train)
    move_file(dfSnowy, s, classifier, is_train)


def severity_divider(df, classifier, is_train):

    # selecte rows that apply to the condition both yes/no
    dfZero = df[df[classifier] == 0]
    dfOne = df[df[classifier] == 1]
    dfTwo = df[df[classifier] == 2]
    dfThree = df[df[classifier] == 3]
    dfFour = df[df[classifier] == 4]

    # store selection in variables
    zero = 0
    one = 1
    two = 2
    three = 3
    four = 4

    move_file(dfZero, zero, classifier, is_train)
    move_file(dfOne, one, classifier, is_train)
    move_file(dfTwo, two, classifier, is_train)
    move_file(dfThree, three, classifier, is_train)
    move_file(dfFour, four, classifier, is_train)


def move_file(df, selected, classifier, is_train):

    sub_folder_dir = os.path.join(src_folder_path, classifier)
    create_directory(sub_folder_dir)

    ### Create a file moving system that does the image division as we discussed with Robert
    if is_train is True:

        df = df_shuffle(df)
        train_df, val_df = df_split(df)

        sub_division_train_folder_dir = os.path.join(src_folder_path, classifier, 'train')
        create_directory(sub_division_train_folder_dir)
        selection_train_folder_dir = os.path.join(sub_division_train_folder_dir, selected + classifier)
        create_directory(selection_train_folder_dir)

        # copy images from image folder to training image destination folder
        copy_images(train_df, selection_train_folder_dir)

        sub_division_valid_folder_dir = os.path.join(src_folder_path, classifier, 'val')
        create_directory(sub_division_valid_folder_dir)
        selection_val_folder_dir = os.path.join(sub_division_valid_folder_dir, selected + classifier)
        create_directory(selection_val_folder_dir)

        # copy images from image folder to validation image destination folder
        copy_images(val_df, selection_val_folder_dir)

    else:
        sub_division_test_folder_dir = os.path.join(src_folder_path, classifier, 'test')
        create_directory(sub_division_test_folder_dir)
        selection_test_folder_dir = os.path.join(sub_division_test_folder_dir, selected + classifier)
        create_directory(selection_test_folder_dir)

        # copy images from image folder to test image destination folder
        copy_images(df, selection_test_folder_dir)



def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)


#shuffles dataFrame
def df_shuffle(car_df):

    car_df = sku.shuffle(car_df)
    return car_df


#split dataset to 80% training and 20% training dataset
def df_split(car_df):

    training_image_count = int(len(car_df)*0.8)

    training_car_df = car_df[0:training_image_count]
    validation_car_df = car_df[training_image_count:]

    return training_car_df, validation_car_df


def copy_images(df, dst_directory):

    for row_index, row in df.iterrows():
        imageName = row['ImageFilename']
        src_dir = os.path.join(src_folder_path, imageName)
        dst_dir = os.path.join(dst_directory, imageName)
        shutil.copyfile(src_dir, dst_dir)


def main():
    #reads the data from a csv file
    training_df = csv_to_dataFrame(training_csv_src)
    testing_df = csv_to_dataFrame(testing_csv_src)


    #ask me which classification I want to work on
    print("Select one of the classifier below **case-sensitive")
    print(list(training_df.columns.values)[3:16])
    selected = input()

    while selected != 'GlassDamage' and selected != 'WaterDamage' and selected != 'BumperDamage' \
        and selected != 'RolledOver' and selected != 'SideDoorDamage' and selected != 'SlidOffRoad' \
        and selected != 'Location' and selected != 'Time' and selected != 'Weather' and selected != 'RoadCondition' \
        and selected != 'Bumps&Dings' and selected != 'EngineDamage' and selected != 'Severity':

        print("Select one of the classifier below **case-sensitive")
        print(list(training_df.columns.values)[3:16])
        selected = input()

    process_data(selected, training_df, testing_df)


if __name__ == "__main__":
    main()