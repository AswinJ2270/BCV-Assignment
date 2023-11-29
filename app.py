import streamlit as st
import cv2
import numpy as np

# Function to perform Canny edge detection
def canny_edge_detection(image):
    edges = cv2.Canny(image, 50, 150)
    return edges

# Function to detect lines using Hough transform
def line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    return lines

# Function to detect corners using Harris corner detection
def harris_corner_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    return corners

# Function to detect keypoints using Hessian Affine
def hessian_affine_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hessian = cv2.xfeatures2d.SIFT_create()
    keypoints, _ = hessian.detectAndCompute(gray, None)
    return keypoints

# Function to perform Laplacian of Gaussian (LOG)
def laplacian_of_gaussian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    log = cv2.GaussianBlur(gray, (3, 3), 0)  # Apply Gaussian blur
    log = cv2.Laplacian(log, cv2.CV_64F)
    log = np.uint8(np.absolute(log))
    return log

# Function to perform Difference of Gaussians (DOG)
def difference_of_gaussians(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dog = cv2.GaussianBlur(gray, (5, 5), 0) - cv2.GaussianBlur(gray, (3, 3), 0)
    return dog

# Streamlit UI
st.title("Computer Vision Techniques Demo")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Display the original image
    st.image(image, caption="Original Image", use_column_width=True)

    # Canny edge detection
    edges = canny_edge_detection(image)
    st.image(edges, caption="Canny Edge Detection", use_column_width=True)

    # Line detection
    lines = line_detection(image)
    line_image = np.copy(image)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        st.image(line_image, caption="Line Detection", use_column_width=True)

    # Harris corner detection
    corners = harris_corner_detection(image)
    corner_image = np.copy(image)
    corner_image[corners > 0.01 * corners.max()] = [0, 0, 255]
    st.image(corner_image, caption="Harris Corner Detection", use_column_width=True)

    # Hessian Affine detection
    keypoints = hessian_affine_detection(image)
    affine_image = cv2.drawKeypoints(image, keypoints, None)
    st.image(affine_image, caption="Hessian Affine Detection", use_column_width=True)

    # Laplacian of Gaussian (LOG)
    log = laplacian_of_gaussian(image)
    st.image(log, caption="Laplacian of Gaussian (LOG)", use_column_width=True)

    # Difference of Gaussians (DOG)
    dog = difference_of_gaussians(image)
    st.image(dog, caption="Difference of Gaussians (DOG)", use_column_width=True)