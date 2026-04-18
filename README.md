# Precision Liver Tumour Segmentation in CT Scans using a modified SegNet Architecture

This project implements semantic segmentation for liver tumour detection using the SegNet architecture, a convolutional encoder-decoder network designed for pixel-wise classification. The system includes image preprocessing, deep learning model inference, and web-based visualization.

---

## Project Structure

C:.
├───images
├───instance
├───processed
├───ss
├───static
├───templates
├───uploads
└───__pycache__


---

## Features

- Deep learning-based segmentation using SegNet
- Support for uploading and segmenting new images
- Web-based interface (Flask/Django compatible)
- Preprocessing pipeline and result visualization
- Structured folders for datasets, outputs, and web content

---

## How to Run

1. **Clone or Download the repository**
   ```bash
   git clone https://github.com/your-username/liver-segnet.git
   cd liver-segnet
   or
   Download as a zip.

2. **Install Datasets**
  Download atleast any 2 or more from this link https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/ according to your system capabilities and change the links in the pr.py program.

3. **Install Libraries**
  Numpy, Pandas, Matplotlib.pyplot, PyDicom, Tensorflow, Werkzeug, Flask, FlaskSQLAlchemy, PIL, cv2    

4. **Run the web server**
  python app.py

5. **Access the application**
  Open your browser and go to http://127.0.0.1:5000

## Upload Instructions
  Place new input images in the uploads/ folder using the UI.
  Outputs will be saved in the processed/ folders depending on the processing type.

## Further Instructions
  When downloaded, if the HTML files and CSS files are not in separate files templates and static, respectively, please create folders and put those files there.
  Also, if you have GPU or any better optimization dont hesitate to use them.
