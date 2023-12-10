<br>

# <div align="center">Permeability Prediction using Vision Transformers</div>
Implementation of Swin-Transformer based permeability prediction on 3D object.

## List of Contents
<li><a href="#introduction"> Introduction</a></li>
<li><a href="#prerequisites"> Prerequisites</a></li>
<li><a href="#folder-structure"> Folder Structure</a></li>
<li><a href="#dataset-preparation"> Dataset Preparation</a></li>
<li><a href="#proposed-framework"> Proposed framework</a></li>
<li><a href="#experiment-result"> Experiment result</a></li>
<li><a href="#acknowledgements"> Acknowledgements</a></li>
<li><a href="#contributors"> Contributors</a></li>

<h2 id="introduction"> Introduction</h2>
  
Accurate permeability prediction remains pivotal in understanding fluid flow in porous media, influencing crucial operations across petroleum engineering, hydrogeology, and related fields. Traditional approaches, while robust, often grapple with the inherent heterogeneity of reservoir rocks. With the advent of deep learning, Convolutional Neural Networks (CNNs) have emerged as potent tools in image-based permeability estimation, capitalizing on micro-CT scans and digital rock imagery. This paper introduces a novel paradigm, employing Vision Transformers (ViTs) - a recent advancement in computer vision - for this crucial task. ViTs, which segment images into fixed-sized patches and process them through transformer architectures, present a promising alternative to CNNs. We present a methodology for implementing ViTs for permeability prediction, results from diverse rock samples, and a comparison against conventional CNNs. The prediction results suggest that, with adequate training data, ViTs can      match or surpass the predictive accuracy of CNNs, especially in rocks exhibiting significant heterogeneity. This study underscores the potential of Vision Transformers as an innovative tool in permeability prediction, paving the way for further research and integration into mainstream reservoir characterization workflows.

<h2 id="prerequisites"> Prerequisites</h2>

The following open source packages are mainly used in this project:
* Numpy
* Pandas
* Matplotlib
* Scikit-Learn
* PyTorch

Please also install other required packages when there is any missing (see detailed package uses in .py files)

<h2 id="folder-structure"> Folder Structure</h2>

    code
    .
    ├── src
      ├── model
        ├── vit_model.py
      ├── utils
        ├── my_dataset.py 
        ├── utils.py   
      ├── data_preprocess.ipynb
      ├── performance_eval.ipynb
      ├── predict.py
      ├── train.py
    
<h2 id="dataset-preparation"> Dataset Preparation</h2>

Data source: https://www.digitalrocksportal.org/projects/372

Features used to characterize the geometry of the void space are studied to predict permeability. These features include single phase mfp, electrical properties elec_uz, and geometric properties like MIS_3D, e_domain, tOf_L, tOf_R. These features are explained as follows.

**Electrostatic simulations: elec_uz**

Quantification of electrical behavior in porous media has supported advancements in petroleum reservoir characterization, CO2 monitoring in carbon capture and storage, hydrogeology, mineral exploration, and battery development. In these composite systems, electrical conductivity measurements aid in inferring the composition of the material and its phase distributions. For example, in petroleum systems, well-bore resistivity (reciprocal of conductivity) measurements are commonly used to estimate the amount of oil in place in the reservoir rock.

**elec_uz.**
Electrical conductivity is a fundamental property of a material that quantifies how strongly it conducts electric current, where high conductivity values mean that the material readily allows current to flow. Similar to permeability, the overall electrical response of subsurface geosystems is subject to rock formation processes and subsequent diagenesis. The conductivity is primarily impacted by the topology of the conductive phase structures. Specifically, conductivity measurements capture the effects of the sinuous transport path of the connected pore space (tortuosity) and variations in the cross-sectional area of the conducting paths (constriction factor). Heterogeneities created by these processes create conductive pathways on a range of length scales similar to that of fluid flow. However, behavior at the nano- and micron-scales arguably has a more profound impact on the macroscopic (regional scale) response for electrical properties than for fluid flow. Therefore, geometric characterization of these small-scale features is crucial for inferring electrical properties on larger scales.

**Geometrical features: MIS_3D, e_domain, tOf_L, tOf_R**
Binary images of porous materials are an important input for applications like direct simulation of physical processes. But, a 3D binary image by itself provides limited information about its overall geometric characteristics. There are many metrics that are commonly computed to characterize the structure of binary images of porous material. In this dataset, there are 10 geometrical features from each binary images described in the previous section and 4 of them are finally chosen in model. These features represent different aspects of the local and global topology of the original structure. These features serve as proxies for better descriptors of binary images of porous media (pore size distribution, tortuosity, local porosity), which are often used to describe sample populations. Furthermore, these features have been used as inputs for machine learning models to study a wide variety of relationships between structure and bulk properties of porous media. The features are grouped in the following categories:

**tOf_L, tOf_R.**
They are **Time of flight maps from left to right**. We used the fast marching algorithm to compute the shortest distance of all the points of the domain to a plane source (In this case, both of the the XY-planes at the first and last slice, individually). This method solves the boundary value problem of the Eikonal equation. The output provides a 3D map which (1) explains how tortuous a path is (or how much a path deviates from a straight line) in the z-direction, (2) conversely also highlights the easiest paths (or highways) for flow, and (3) describes how connected the domain is overall.

![image](https://github.com/LeeGorden/PoreFlow/assets/72702872/b7d4518c-b8c8-4444-9af1-e802f92f5dbb)

All of the 6 features described above have a dimension of 256 × 256 × 256. Then, these 6 features are concatenated into a six-channel 3D cube as data input, shown as follows.

![image](https://github.com/LeeGorden/PoreFlow/assets/72702872/57295728-3adb-4629-8ade-d1cf7f23b181)

<h2 id="proposed-framework"> Proposed Framework</h2>
  
For predicting permeability, in the model structure section, the encoder of Swin Transformer proposed by Liu, Z. et al., (2021) is chosen. The encoder of Swin Transformer is chosen because traditional transformer has limitations regarding the length of sequences, in this case the number of 3D points in the 3D cube, especially when considering the resources consumption will increase significantly when object dimension increases from 2D to 3D. The idea of using successive window and shifted-window successively to aggregate self-attention-based information instead of global self-attention-based information aggregation is the core of reducing complexity.

The Swin Transformer Encoder backbone has been applied in solving 3D object detection, segmentation, and classification. (Cao, H.  et al. 2022; Tang, Y. et al. 2022; Hatamizadeh, A. et al. 2022). We inherit the encoder structure of 3D Medical Image Segmentation conducted by Hatamizadeh, A. et al. (2022), as shown in the following figure. We use the patch size of  8 × 8 × 8 instead of the original patch size of 4 × 4 × 4 in order to absorb more diverse geological information.
  
![image](https://github.com/LeeGorden/PoreFlow/assets/72702872/fbf7d647-95b9-4064-8e92-2a404b15b84c)

<h2 id="experiment-result"> Experiment result</h2>

- Model performance:
  
  ![image](https://github.com/LeeGorden/PoreFlow/assets/72702872/05d700ce-d1c4-4f12-9f5d-275daffd8c33)

- Ablation analysis:

  ![image](https://github.com/LeeGorden/PoreFlow/assets/72702872/4275aa67-aee1-4463-8a45-48c204b4d763)

<h2 id="acknowledgements"> Acknowledgements</h2>

<h2 id="contributors"> Contributors</h2>

  <b>Cenk Temizel</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/cenktemizel">@Cenk Temizel</a> <br>

  <b>Uchenna Odi</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/chempeodi">@Uchenna Odi</a> <br>

  <b>Kehao Li</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/LeeGorden">@Kehao Li</a> <br>

  <b>Javier E.Santos</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/je-santos">@Javier E.Santos</a> <br>

<br>
