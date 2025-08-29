<p align="center">
  <img src="other/logo_unipd.png" alt="" height="70"/>
</p>

# From Pixels to Points: AI-Powered Tennis Match Insights

</div>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" height="25"/>
  <img alt="Latex" src="https://img.shields.io/badge/Latex-008080?style=for-the-badge&logo=latex&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Visual Studio Code" src="https://img.shields.io/badge/Visual Studio Code-007ACC?logo=VisualStudioCode&logoColor=white&style=for-the-badge" height="25"/>
  <img alt="Overleaf" src="https://img.shields.io/badge/Overleaf-47A141?style=for-the-badge&logo=overleaf&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Google Colab" src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?logo=Jupyter&logoColor=white&style=for-the-badge" height="25"/>
  <img alt="Git" src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white&logoSize=auto" height="25"/>
  <img alt="GitHub" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white&logoSize=auto" height="25"/>
<p>

</div>
<p align="center">
  <img src="other\output_video1012.gif" alt=""/>
</p>
</div>

## Overview

This repository contains the project for the "Vision and Cognitive Systems" course.

This project provides a comprehensive suite of computer vision tools for detailed analysis of tennis matches. Our system is designed to extract key insights from tennis videos by detecting and tracking players and the ball, identifying important events like racket hits and ball bounces, and analyzing game dynamics. We also developed a novel method to predict player score probabilities based on a ball and score heatmap

The core contributions of this project include:

* **Ball Bounce Detection**: We developed a new deep learning method that treats bounce detection as a visual pattern recognition problem. Our approach uses a lightweight Convolutional Neural Network (CNN) combined with DBSCAN clustering to achieve an $84$% accuracy in detecting true bounces within a 2-frame tolerance.
* **Racket Hit Detection**: We created an audio-visual approach that combines audio signals with visual cues to detect when a player hits the ball. This method demonstrated an $84.74$% accuracy within a 6-frame range.
* **Score Probability Prediction**: Our system introduces a computer vision-based method to predict shot-by-shot score probabilities by analyzing the ball's estimated landing position and the opponent's location.

Our work represents a significant step forward in automated sports analysis, providing tools that can give viewers and analysts much deeper insights into tennis matches.

</div>

## Results

### Racket Hit Detection

Our model achieved an **84.74% accuracy** in detecting racket hits within 6 frames of the actual event and a **72.11% accuracy** within a 2-frame tolerance. The system missed some hits in $22.86$% of the videos and detected false positives in $11.43$% of the videos.

</div>

### Ball Bounce Detection

We developed a custom dataset for bounce detection and formulated the problem as a binary classification task. The model's performance was evaluated using precision, recall, and F1-score. Our CNN model, when combined with DBSCAN clustering for exact frame prediction, achieved an **84% accuracy** in detecting true bounces with a tolerance of $\pm2$ frames. The table below summarizes the model's performance metrics.

| Precision | Recall | F1 Score |
|:---:|:---:|:---:|
| $0.7429$ | $0.8966$ | $0.8125$ |

</div>

### Courtline Prediction

Using a fine-tuned ResNet-34 model, we predicted the coordinates of 14 key court points. The model was trained with MSE loss and achieved a total loss of **1.16 on the training set** and **9.26 on the validation set**.

</div>

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/danielevirzi/Tennis_Sport_Analysis.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python main.py
```

## Project Structure

* `main.py`: Entry point for running the analysis pipeline.
* `data_scrapert/`: Scripts for scraping video data.
* `ball_landing/`: Code for ball landing detection.
* `court_line_detector/`: Code for detecting and predicting court lines.
* `data/`: Contains datasets used for training and evaluation.
* `mini_court/`: Code for court detection and line prediction.
* `models/`: Pre-trained models and architecture definitions.
* `output/`: Stores output videos and results.
* `other/`: Contains images, logos, and other media files.
* `trackers/`: Player and ball tracking algorithms.
* `training/`: Scripts and notebooks for training models.
* `utils/`: Utility functions for data processing and visualization.

## Contributors

* [Daniele Virzì](https://github.com/danielevirzi)
* [Marlon Helbing](https://github.com/maloooon)
* [Alberto Calabrese](https://github.com/Albi1999)

For contributions, suggestions, or inquiries, feel free to reach out.

## License

This project is open-source under the MIT License. See `LICENSE` for more information.
