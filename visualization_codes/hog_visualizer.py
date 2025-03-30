from skimage import feature, io, color
import matplotlib.pyplot as plt

# Read the image
image = io.imread('image.png')

# Convert image to grayscale
gray_image = color.rgb2gray(image)

# Extraction of HOG feature
hog_features, hog_image = feature.hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
                                      cells_per_block=(1, 1), visualize=True)

# Plot the original image and the HOG image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(gray_image, cmap=plt.cm.gray)
ax1.set_title('Original')

ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('HOG Features')

plt.show()
