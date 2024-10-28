from PIL import Image
import itertools


def pixelate_image(image_path):
    # Open the image
    pixel_size = 16
    img = Image.open(image_path)

    # Calculate dimensions for the pixelation
    width, height = img.size
    new_width = width // pixel_size
    new_height = height // pixel_size

    # Resize the image to small size
    small_img = img.resize((new_width, new_height), resample=Image.BILINEAR)

    # Scale back up using NEAREST to introduce pixelation effect
    pixelated_img = small_img.resize(img.size, resample=Image.NEAREST)

    return pixelated_img


def images(image):
    image_path = image
    pixel_size = 16  # Assuming we want 16x16 big pixels
    pixelated_image = pixelate_image(image_path)
    pixelated_image.save(f"media/pixels/pixel.jpg")  # Save the pixelated image


# Define the list of alphabets and numbers to use for naming colors
color_labels = list(itertools.product(['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y'], range(1, 10)))


# Function to convert RGB to hexadecimal
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


# Function to get color name from RGB using predefined labels
def get_color_name(rgb):
    hex_color = rgb_to_hex(rgb)
    try:
        color_index = sum(rgb) % len(color_labels)  # Generate index based on RGB sum
        color_name = f"{color_labels[color_index][0]}{color_labels[color_index][1]}"
    except ValueError:
        color_name = "Unknown"  # Handle case where color name cannot be determined
    return color_name


# Function to cut image into tiles and save colors as JSON
def cut_image_and_save_colors_as_json(image_path, tile_width, tile_height):
    # Open the image
    img = Image.open(image_path)

    # Get the dimensions of the image
    img_width, img_height = img.size

    # Calculate number of rows and columns
    rows = img_height // tile_height
    cols = img_width // tile_width

    # List to store color data of each tile
    tiles_colors = []

    # Counter for IDs
    id_counter = 1

    # Cut the image into tiles
    for r in range(rows):
        for c in range(cols):
            box = (c * tile_width, r * tile_height, (c + 1) * tile_width, (r + 1) * tile_height)
            tile = img.crop(box)
            tile_colors_rgb = list(tile.getdata())

            # Prepare list of color dictionaries with hex code and color name
            tile_colors_data = []
            for color_rgb in tile_colors_rgb:
                hex_code = rgb_to_hex(color_rgb)
                color_name = get_color_name(color_rgb)
                tile_colors_data.append({
                    'hex_code': hex_code,
                    'color_name': color_name
                })

            # Divide tile colors into groups of 9
            groups_of_9 = [tile_colors_data[i:i + 9] for i in range(0, len(tile_colors_data), 9)]

            # Prepare groups for the tile
            groups_data = [{'id': id_counter + idx, 'colors': group} for idx, group in enumerate(groups_of_9)]

            # Append tile data to tiles_colors
            tiles_colors.append({
                'groups': groups_data
            })

            # Update ID counter
            id_counter += len(groups_of_9)

    print(f"Image cut into {rows} rows and {cols} columns of size {tile_width}x{tile_height}.")
    return tiles_colors
