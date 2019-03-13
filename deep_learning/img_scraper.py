import urllib.request
import os.path

def download_image(url, dir_path):

    for i in range(1, 800):

        img_num = '00'

        if 0 < i < 10:
            img_num += '00' + str(i)
            # print(img_num)

        elif 10 <= i < 100:
            img_num += '0' + str(i)
            # print(img_num)

        elif 100 <= i < 1000:
            img_num += str(i)
            # print(img_num)

        # creates the number 00001 ~ 00799

        img_type = ['.jpg', '.jpeg', '.png', '.JPG', '.gif', '.GIF']

        for type in img_type:

            file_name = 'car-accident-' + img_num + type
            file_path = os.path.join(dir_path, file_name)
            #print(file_path)

            temp_url = os.path.join(url, file_name)
            #print(temp_url)

            try:
                urllib.request.urlretrieve(temp_url, filename=file_path)
                print('Success', temp_url)

            except:
                # print('Fail', file_name)
                pass



url = "http://oltivainsurance8754.blob.core.windows.net/car-dataset/"
local_dir_path = '/Users/Koshin/Pictures/car_accident_imgs/'

download_image(url, local_dir_path)

