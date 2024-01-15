import numpy as np
import cv2

def convert_yuv_to_rgb(yuv_frame, width, height):
    # Reshape YUV data based on the video properties
    yuv_frame_reshaped = yuv_frame.reshape((height + height // 2, width))

    # Convert YUV to RGB
    rgb_frame = cv2.cvtColor(yuv_frame_reshaped, cv2.COLOR_YUV2RGB_I420)

    return rgb_frame

def convert_rgb_to_gray(rgb_frame):
    # Convert RGB to grayscale
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

    return gray_frame

def main():
    # Input and output file paths
    input_yuv_file = '/home/idm/test_convert/Reeds_4976x3456.yuv'
    output_yuv_file = '/home/idm/test_convert/ReedsGray_4976x3456.yuv'

    # Video properties (width, height, frames)
    width = 4976
    height = 3456
    num_frames = 1

    # Read the YUV video
    yuv_data = np.fromfile(input_yuv_file, dtype=np.uint8)

    # Reshape the YUV data based on the video properties
    yuv_frames = yuv_data.reshape((num_frames, height + height // 2, width))

    # Iterate through each frame, convert to grayscale, and save to a new YUV file
    with open(output_yuv_file, 'wb') as output_file:
        for yuv_frame in yuv_frames:
            # Convert YUV to RGB
            rgb_frame = convert_yuv_to_rgb(yuv_frame, width, height)

            # Convert RGB to grayscale
            gray_frame = convert_rgb_to_gray(rgb_frame)

            # Convert grayscale back to RGB for YUV conversion
            rgb_frame_from_gray = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Convert RGB to YUV
            yuv_frame_out = cv2.cvtColor(rgb_frame_from_gray, cv2.COLOR_RGB2YUV_I420)

            # Write the YUV frame to the output file
            output_file.write(yuv_frame_out.tobytes())

    print("Conversion complete. Grayscale YUV video saved to", output_yuv_file)

if __name__ == "__main__":
    main()
