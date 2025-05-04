# Text Recognition Application

A comprehensive Python application for extracting and analyzing text, numbers, and words from images using Optical Character Recognition (OCR). This application provides a user-friendly interface with advanced features for image processing, text extraction, and data analysis.

![Text Recognition App](https://github.com/yourusername/text-recognition-app/raw/main/screenshots/app_screenshot.png)

## Features

### Core Functionality
- **Text Extraction**: Extract all text, numbers, and words from images
- **Tabbed Results**: View extracted content in organized tabs (All Text, Numbers, Words)
- **Recognition History**: Save and browse through previous recognition sessions

### Image Processing
- **Contrast Enhancement**: Improve text visibility in low-contrast images
- **Noise Reduction**: Clean up noisy images for better recognition accuracy
- **Image Deskewing**: Automatically straighten tilted text for improved results
- **Preview**: See the image before processing

### Advanced Features
- **Multi-language Support**: Recognize text in English, French, Spanish, German, Arabic, and more
- **Camera Integration**: Capture images directly from your webcam
- **Batch Processing**: Process multiple images at once with progress tracking
- **Export Options**: Save results as TXT or CSV files
- **Text-to-Speech**: Listen to extracted text through audio playback
- **Confidence Threshold**: Filter results based on recognition confidence
- **Search Functionality**: Search through your recognition history

### Analysis Tools
- **Statistics Dashboard**: Visualize recognition data with interactive charts
- **Most Common Words/Numbers**: See frequency analysis of extracted content
- **Processing Metrics**: Track total images processed and text extracted

### Customization
- **OCR Engine Settings**: Fine-tune the recognition engine parameters
- **Page Segmentation Modes**: Control how the engine analyzes page layout
- **Character Whitelist**: Specify which characters to recognize
- **Custom Confidence Thresholds**: Adjust sensitivity of text detection

## Installation

### Prerequisites

- Python 3.7 or higher
- Tesseract OCR engine (v4.0+ recommended)
- Webcam (optional, for camera capture feature)

### Step 1: Install Tesseract OCR

#### Windows
1. Download the installer from [UB Mannheim's GitHub repository](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and note the installation path (default is `C:\Program Files\Tesseract-OCR`)
3. Make sure to check "Add to PATH" during installation
4. For additional languages, select them during installation

#### macOS
```bash
brew install tesseract
# For additional languages
brew install tesseract-lang
