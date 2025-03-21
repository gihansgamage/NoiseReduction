import cv2
import matplotlib.pyplot as plt

def noise_reduction(image_path):
    img_noise = cv2.imread(image_path)
    img_noise_rgb = cv2.cvtColor(img_noise, cv2.COLOR_BGR2RGB)

    gaussian_blur = cv2.GaussianBlur(img_noise, (5,5),0)
    gaussian_rgb = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB)

    median_blur = cv2.medianBlur(img_noise, 5)
    median_rgb = cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB)

    bilateral_filter = cv2.bilateralFilter(img_noise, 9,100,100)
    bilateral_rgb = cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB)

    nonlocal_mean = cv2.fastNlMeansDenoisingColored(img_noise, None, 10, 10 ,7,21)
    nonlocal_rgb = cv2.cvtColor(nonlocal_mean,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14,8))

    tup = [("Noise Image", img_noise_rgb),
           ("Gaussian Image", gaussian_rgb),
           ("Median Image", median_rgb),
           ("Bilateral Image", bilateral_rgb),
           ("Nonlocal image", nonlocal_rgb)]
    
    for t in range(len(tup)-1):
        plt.subplot(2,3,t+1)
        plt.title(tup[t][0])
        plt.imshow(tup[t][1])
        plt.axis("off")
    plt.show()


noise_reduction("flower_0.10_noisy.jpg")
noise_reduction("noise1.png")
