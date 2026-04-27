# CNN-Based Cross-Algorithm Steganalysis Demo
Interactive demo for my RGU CM4134 Capstone Project investigating cross-algorithm generalisation of CNN-based steganalysis.

# Project Overview
This app demonstrates a replication of the Xu et al. (2016) CNN architecture trained across multiple steganographic embedding algorithms: HUGO, WOW, S-UNIWARD, and HILL — on the BOSSbase 1.01 dataset at 0.4 bpp.

## Models Available
- Matched Baseline: S-UNIWARD and HILL
- Best 1-against-3: HILL trained
- Best 2-against-2: HILL + WOW trained
- Best 3-against-1: S-UNIWARD+HILL+WOW trained

## Usage
Upload a grayscale PNG Image and select a model to classify it as cover or stego.
