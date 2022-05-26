# Altr

Altr is a simple script that generates and places alternate text for images in a static website. It's goal is to improve SEO (search engine optimization) and accessibility for websites with many images.

# Getting started

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yaml
    ```

2. Install the CLIP repository 
    ```
    pip install git+https://github.com/openai/CLIP.git
    ```

3. Download pretrained weights (conceptual_weights.pt) and place them in the weights directory. You can find the [COCO](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view?usp=sharing) and [Conceptual Captions](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view?usp=sharing) pretrained models on Google Drive.

4. Launch your environment with `conda activate understanding` or `source activate understanding`

5. Place files to change in `input` folder

6. Run `python main` to generate captions for files. New files will be placed in `output` folder.
