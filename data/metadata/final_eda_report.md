# SmartDocs AI Phase 6 Exploratory Data Analysis

## Dataset Overview

| Split | Images |
| --- | --- |
| train | 1400 |
| validation | 300 |
| test | 300 |

Total images: **2000**

### Class Counts

| Class | Images |
| --- | --- |
| invoices | 500 |
| forms | 500 |
| resumes | 500 |
| other | 500 |

## Class Distribution

Balance status: **BALANCED**

Percentages are 25.00% per class in each split and in the complete dataset.

## Image Properties

| Property | Observed value |
| --- | --- |
| Dimensions | 224x224 |
| Color modes | RGB |
| File extensions | {'.png': 2000} |
| File size range | 1665 - 109160 bytes |
| Average file size | 31296.62 bytes |

## Visual Analysis

Random seed: **42**. Six samples were selected per class, with two samples from each analyzed split.

Generated visualizations:

- `data/metadata/class_distribution.png`
- `data/metadata/eda_visual_samples.png`
- `data/metadata/classwise_mean_intensity.png`
- `data/metadata/classwise_file_size.png`

## Pixel Statistics

| Statistic | Overall value |
| --- | --- |
| Mean intensity | 78.919533 |
| Standard deviation | 114.713107 |
| RGB means | {'mean_R': 78.919533, 'mean_G': 78.919533, 'mean_B': 78.919533, 'standard_deviation_R': 114.713107, 'standard_deviation_G': 114.713107, 'standard_deviation_B': 114.713107} |
| Blank / dark / bright | 0 / 1 / 1783 |

Thresholds used: `{'pixel_scale': '8-bit RGB intensity, 0-255', 'blank_image': 'all RGB pixel values equal 0', 'dark_image_mean_intensity_less_than': 32, 'bright_image_mean_intensity_greater_than': 224}`

## Class-wise Comparison

| Class | Mean intensity | Mean file size (bytes) |
| --- | ---: | ---: |
| invoices | 79.583443 | 29241.33 |
| forms | 79.522463 | 29756.55 |
| resumes | 79.235665 | 36985.08 |
| other | 77.336559 | 29203.52 |

Highest average brightness: **['invoices']**  
Lowest average brightness: **['other']**  
Largest average file size: **['resumes']**  
Smallest average file size: **['other']**

These are descriptive comparisons only and do not indicate class quality or model performance.

## Dataset Quality Checks

| Check | Result |
| --- | --- |
| Unreadable images | 0 |
| Unsupported files | 0 |
| Missing directories | 0 |
| Inconsistent dimensions | False |
| Inconsistent color modes | False |
| Duplicate/content issues | 0 |

## Final Observations

- The dataset contains 2000 images: 1400 train, 300 validation, and 300 test.
- Class distribution is balanced across the complete dataset and within each split.
- All readable images have consistent dimensions and color mode.
- The observed image properties are ['224x224'] with color mode(s) ['RGB'] and extension counts {'.png': 2000}.
- The overall mean pixel intensity is 78.919533 with standard deviation 114.713107.
- Blank, dark, and bright image counts are 0, 1, and 1783 under the documented thresholds.

## Summary

**Phase 6 EDA complete: True**

All required checks pass only when this value is `True`.
