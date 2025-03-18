import cv2 as cv
import numpy as np


def read_flaser_line(line):
    tokens = line.split()

    # Check if the line has enough data and doesn't contain unexpected strings
    if len(tokens) < 2 or not tokens[1].replace('.', '', 1).isdigit():
        return None  # Skip invalid lines

    # Convert all valid numeric values
    data = list(map(float, tokens[1:]))
    
    num_readings = int(data[0])  # Number of LiDAR readings

    # Ensure that there are enough values to extract LiDAR distances and robot pose
    if len(data) < num_readings + 7:  # 7 = 1 (num_readings) + num_readings + 6 (robot pose data)
        return None  # Skip incomplete lines

    ranges = np.array(data[1:num_readings + 1])  # Extract LiDAR distances
    robot_x, robot_y, robot_theta = data[-6], data[-5], data[-4]  # Extract robot pose

    return ranges, robot_x, robot_y, robot_theta

def conv_cartesian(data,theta,x,y,scale=0.1,origin=(316,139)):
    coords = []
    num_points = len(data)

    # Generate angles from -π/2 to π/2
    angles = np.linspace(-np.pi/2, np.pi/2, num_points)

    for d, angle_offset in zip(data, angles):
        angle = theta + angle_offset  # Adjust with robot heading
        px, py = x + d * np.cos(angle), y + d * np.sin(angle)
        coords.append((int((px / scale) + origin[0]), int((py / scale) + origin[1])))

    return coords


def preprocessing_map(img_path):
    map_image = cv.imread(img_path)
    hsv = cv.cvtColor(map_image, cv.COLOR_BGR2HSV)
    lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
    mask = cv.inRange(hsv, lower_green, upper_green)
    map_image[mask > 0] = [255, 255, 255]
    cv.imwrite("processed_map.png",map_image)

def correlative_scan_matching(map_img, scan_points):
    """Find the best match for the scan on the map (simple version)."""
    best_x, best_y = 0, 0
    best_score = -np.inf
    h, w = map_img.shape
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            test_points = scan_points + np.array([dx, dy])
            score = np.sum(map_img[np.clip(test_points[:, 1].astype(int), 0, h-1),
                                   np.clip(test_points[:, 0].astype(int), 0, w-1)])
            if score > best_score:
                best_score = score
                best_x, best_y = dx, dy
    return best_x, best_y

def main():
    """Main function to process data and localize the robot."""
    preprocessing_map("videos/agv.png")  # Process and save the map
    map_img = cv.imread("processed_map.png", cv.IMREAD_GRAYSCALE)  # Reload in grayscale

    with open("videos/aces.clf.txt", "r") as file:
        for line in file:
            if line.startswith("FLASER"):
                ranges, x, y, theta = read_flaser_line(line)
                scan_points = conv_cartesian(ranges,theta,x,y)
                dx, dy = correlative_scan_matching(map_img, scan_points)
                print(f"Best position adjustment: ({dx}, {dy})")

if __name__ == "__main__":
    main()


