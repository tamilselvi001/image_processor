# Image Processing
https://imageproceapprgit.streamlit.app/
A lightweight local image-processing application built with Python, Streamlit, OpenCV, NumPy, Pillow and Matplotlib.

## Features

- Pixel-level image processing
- Grayscale, negative and thresholding
- Histogram visualization
- Brightness, contrast and gamma correction
- Sharpening and Laplacian sharpening
- Gaussian, median, average and bilateral filtering
- Low-pass and high-pass spatial filtering
- Roberts, Prewitt and Sobel gradients
- Gradient magnitude and direction
- Roberts, Prewitt, Sobel, Laplacian and LoG edge detection
- Adjustable Canny edge detection
- Contour and basic shape detection
- Original/result comparison and image download
- Dark and light themes

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

No backend, database, API or cloud service is required.

## Structure

```text
Image_Processing/
├── app.py
├── requirements.txt
├── README.md
├── processors/
│   ├── spatial.py
│   ├── enhancement.py
│   ├── noise_removal.py
│   ├── filters.py
│   ├── gradients.py
│   ├── edge_detection.py
│   ├── canny.py
│   └── shapes.py
├── ui/
│   ├── components.py
│   ├── image_display.py
│   └── theme.py
└── assets/
```
