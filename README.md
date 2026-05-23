# Brain Tumor MRI Classification Using Deep Learning

This repository implements a multi-class deep learning pipeline to identify and categorize brain tumors from magnetic resonance imaging (MRI) scans. Leveraging transfer learning with a ResNet-18 architecture, the system classifies brain scans into four distinct categories: Glioma, Meningioma, Pituitary, or No Tumor. 

The goal is to develop a reliable, automated assistance tool for computer-aided diagnosis in medical imaging workflows.


## Model Architecture & Preprocessing

The classification engine relies on a pre-trained **ResNet-18 backbone**, modified to replace the final linear layer to align with our four-class target space. 

### Ingestion & Image Transformations
To prepare medical images for deep learning inference, scans are processed through a structured normalization and interpolation pipeline:
* **Resolution Standardization:** Raw MRI scans vary in resolution and aspect ratio. Images are programmatically resized to a uniform $224 \times 224$ pixel input layer.
* **Tensor Conversion:** Image structures are scaled from standard pixel ranges down to tensor values falling between $0.0$ and $1.0$.
* **Distribution Scaling:** Pixels undergo standard ImageNet mean and standard deviation scaling ($[\mu_R, \mu_G, \mu_B] = [0.485, 0.456, 0.406]$, $[\sigma_R, \sigma_G, \sigma_B] = [0.229, 0.224, 0.225]$) to stabilize initial gradients during fine-tuning.



## Model Training & Hyperparameters

The network was trained end-to-end using cross-entropy loss optimization. We partitioned the dataset into an 80% training split and a 20% validation/testing split to measure generalization accuracy.

* **Optimizer:** Adam
* **Learning Rate ($\eta$):** 0.001
* **Batch Size:** 32
* **Training Epochs:** 25



## Evaluation Metrics & Diagnostics

The classification engine maps its predictions using a normalized **Confusion Matrix** to ensure diagnostic reliability. Rather than looking solely at raw accuracy numbers, this approach allows for clear performance evaluations across each individual tumor pathology.

* **Diagnostic Transparency:** By evaluating the true label boundaries against predicted coordinates via Seaborn heatmaps, we can monitor specific error profiles—such as distinguishing challenging boundaries between gliomas and meningiomas.
* **Performance Validation:** Normalized row vectors capture explicit classification sensitivity per pathological class.



