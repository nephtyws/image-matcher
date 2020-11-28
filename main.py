import numpy as np
import os
import pandas
import pickle
from PIL import Image
import random
import requests
import scipy.fftpack
import shutil
import time


def color_image_hash(image, binbits=3):
    intensity = np.asarray(image.convert("L")).flatten()
    h, s, v = [np.asarray(v).flatten() for v in image.convert("HSV").split()]

    black_mask = intensity < 256 // 8
    black_fraction = black_mask.mean()

    gray_mask = s < 256 // 3
    gray_fraction = np.logical_and(~black_mask, gray_mask).mean()

    color_mask = np.logical_and(~black_mask, ~gray_mask)
    faint_color_mask = np.logical_and(color_mask, s < 256 * 2 // 3)
    bright_color_mask = np.logical_and(color_mask, s > 256 * 2 // 3)

    color = max(1, color_mask.sum())

    hue_bins = np.linspace(0, 255, 6 + 1)

    if faint_color_mask.any():
        h_faint_count, _ = np.histogram(h[faint_color_mask], bins=hue_bins)

    else:
        h_faint_count = np.zeros(len(hue_bins) - 1)

    if bright_color_mask.any():
        h_bright_count, _ = np.histogram(h[bright_color_mask], bins=hue_bins)

    else:
        h_bright_count = np.zeros(len(hue_bins) - 1)

    max_value = 2 ** binbits
    values = [min(max_value - 1, int(black_fraction * max_value)), min(max_value - 1, int(gray_fraction * max_value))]

    for count in list(h_faint_count) + list(h_bright_count):
        values.append(min(max_value - 1, int(count * max_value * 1. / color)))

    bits = []

    for v in values:
        bits += [v // (2 ** (binbits - i - 1)) % 2 ** (binbits - i) > 0 for i in range(binbits)]

    hash_representation = np.asarray(bits).reshape((-1, binbits)).flatten()

    bit_string = ''.join(str(b) for b in 1 * hash_representation)
    width = int(np.ceil(len(bit_string) / 4))
    return '{:0>{width}x}'.format(int(bit_string, 2), width=width)


def perceptual_image_hash(image, hash_size=8, high_frequency_factor=4):
    image_size = hash_size * high_frequency_factor

    image = image.convert("L").resize((image_size, image_size), Image.ANTIALIAS)
    pixels = np.asarray(image)

    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    dct_flow_frequnecy = dct[:hash_size, :hash_size]

    med = np.median(dct_flow_frequnecy)
    diff = dct_flow_frequnecy > med

    return diff.flatten()


SHOP_NAME = ""

if __name__ == "__main__":
    goods = pandas.read_csv(f"{SHOP_NAME}/goods.csv", usecols=["id", "product_no", "image", "price", "title"])
    image_database = {}
    hash_database = {}
    perceptual_database = []

    # TODO: Read gif and separate it frame by frame?
    print(f"Total {goods.shape[0]} images...")

    for g in goods.iterrows():
        image_path = f"./{SHOP_NAME}/images/{g[1]['id']}.png"
        headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"}

        if not os.path.exists(image_path):
            response = requests.get(g[1]['image'], headers=headers, stream=True)

            # TODO: Create a folder if not exists
            with open(image_path, "wb") as image:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, image)
                print(f"{g[1]['title']}, {g[1]['image']}")
                print(f"Complete to download {g[1]['id']} - {g[1]['title']} : {g[1]['image']}")

            time.sleep(random.randint(0, 2))

        image_database[g[1]['id']] = {
            'product_no': g[1]['product_no'],
            'image': g[1]['image'],
            'price': g[1]['price'],
            'title': g[1]['title'],
        }

    for key in image_database:
        print(image_database[key])

        image_path = f"./{SHOP_NAME}/images/{key}.png"
        image = Image.open(image_path)

        color_hash = color_image_hash(image)
        array = perceptual_image_hash(image)

        image_database[key]['array'] = array

        if color_hash in hash_database:
            hash_database[color_hash].append(image_database[key])

        else:
            hash_database[color_hash] = []
            hash_database[color_hash].append(image_database[key])

        perceptual_database.append(image_database[key])

    with open(f"./{SHOP_NAME}/hash_database.pkl", "wb") as db:
        pickle.dump(hash_database, db)

    with open(f"./{SHOP_NAME}/perceptual_database.pkl", "wb") as db:
        pickle.dump(perceptual_database, db)

    # Retrieve image
    with open(f"./{SHOP_NAME}/hash_database.pkl", "rb") as db:
        hash_database = pickle.load(db)

    with open(f"./{SHOP_NAME}/perceptual_database.pkl", "rb") as db:
        perceptual_database = pickle.load(db)

    image_path = f""
    image = Image.open(image_path)

    color_hash = color_image_hash(image)
    array = perceptual_image_hash(image)

    global_diff, index = 999999999, 999999999

    if color_hash in hash_database:
        if len(hash_database[color_hash]) < 1:
            print(f"Found image in database! {hash_database[color_hash]}")

        else:
            for i, d in enumerate(hash_database[color_hash]):
                diff = np.count_nonzero(d['array'] != array)

                if global_diff >= diff:
                    global_diff = diff
                    index = i

        if global_diff >= 24:
            for i, item in enumerate(perceptual_database):
                diff = np.count_nonzero(item['array'] != array)

                if global_diff >= diff:
                    global_diff = diff
                    index = i

            del perceptual_database[index]['array']
            print(f"[PD] It seems we have found the image in database: {perceptual_database[index]} with diff {global_diff}")

        else:
            del hash_database[color_hash][index]['array']
            print(f"[H] It seems we have found the image in database: {hash_database[color_hash][index]} with diff {global_diff}")

    else:
        for i, item in enumerate(perceptual_database):
            diff = np.count_nonzero(item['array'] != array)

            if global_diff >= diff:
                global_diff = diff
                index = i

        del perceptual_database[index]['array']
        print(f"[P] It seems we have found the image in database: {perceptual_database[index]} with diff {global_diff}")
