
# NeRF (Neural Radiance Fields) research

## TODO

- [ ] setup cuda zesoi and run training there with instant-ngp 

## Log

2022-10-27:

- first result on custom of a caculator: <https://www.youtube.com/watch?v=0SqYMH9wiwg>
- tried to perform training with instant-ngp (instant nerf) but ran out of memory
- installing and preparing starting <https://github.com/nerfstudio-project/nerfstudio>
- reading and note taking: Nerfies: Deformable Neural Radiance Fields
- reading and note taking: Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields

# Nerfies: Deformable Neural Radiance Fields

https://nerfies.github.io/

## Abstract
- deformation field: wrap each observed point into deformation field (MLP) which is also optimized
- coarse-to-fine optimization: start by zeroing out the higher frequency at the start of the optimization (low level details) and the network will learn smooth deformations. Later, introduce higher frequencies
- elastic regulairzation (?)
- two phones: one phone can be used for training while the other can generate test data


## 1. Introduction
"Deformation field is conditioned on a per-image learned latent code" => this suggests that the deformation field varies between each observation

## 3. Deformable Neural Radiance Fields
![](imgs/nerfies-arh.jpg)
- each image has its own latent deformation code ($\omega$) and latent appearance code ($\psi$)
- NeRF isn't querying from the observation frame (classical approach, ($x, y, z$) ), rather, it's querying from the canoncial frame ($x', y', z'$) 
- deformation field is key extension that allows representation of moving subjects
- elastic regularization (3.3), background regularization (3.4), coarse-to-fine (3.5), has to be introduced to avoid under-constrained optimization of deformation field + NeRF

## 3.1 NeRF

NeRF:
* NeRF is a function $F : ({\mathbf x,d,\psi_i}) \rightarrow ({\mathbf c}, \sigma)$ which maps:
  * 3D position ${\mathbf x} = (x,y,z)$ and
  * viewing direction $\mathbf{d} = (\phi, \theta)$
  * to $\mathbf{c} = (r,g,b)$ and
* In pratice, NeRF maps $\mathbf x$ and $\mathbf d$ using a sinusodial positional encoding $\gamma : \mathbb{R^3} \rightarrow \mathbb{R^{3+6m}}$ where $m$ is hyperparameter, total number of frequency bands
* $\gamma(\mathbf{x})=\left(\mathbf{x}, \cdots, \sin \left(2^k \pi \mathbf{x}\right), \cos \left(2^k \pi \mathbf{x}\right), \cdots\right)$

$\psi_i$ latent appearance code exists for each image and it **modulates the color output** to handle interpolation between exposure/white balanace between frames

## 3.2 Neural Deformation Fields
How to allow NeRF to reconstruct non-rigidly deformations? Use canonical template of the scene. To create canonical template they define a mapping $T_i : (x, \omega_i) \rightarrow x'$
- $\omega_i$ is a per-frame learned latetnt deformation code

Problem: rotating a group of points with a translation field requires a different translation for each point. 

<details><summary>
SE(n): proper rigid transformations
</summary>
https://www.wikiwand.com/en/Rigid_transformation

rigid transformation (also called Euclidean transformation or Euclidean isometry) is a geometric transformation of a Euclidean space that preserves the Euclidean distance between every pair of points. Any object will keep the same shape and size after a proper rigid transformation. 

Transformations that also perserve handedness of objects (orientability) are called proper rigid transformation. 

The set of proper rigid transformations is called special Euclidean group, denoted SE(n).
</details>

Solution: use proper rigid transformations in a 3-dimensional Euclidean space, denoted **SE(3)**.

A mapping $W : (x, ωi ) \rightarrow SE(3)$ encodes rigid motion which allows rotation of a set of points which are far away from one another. SE(3) requires only **one rotation parameter** while translation field requires parameter for each point. It's easier to optimize one parameter.

<details><summary>
SE(3) paper snippet
</summary>

![](imgs/nerfies-se3.jpg)
</details>

## 3.3 Elastic regularization

Problem: object is moving backwards and is visually shrinking
Solution: use elastic energy to measure deviation of local deformations from a rigid motion
Takeaway: create jacobian and prenalize the deviation from the closest rotation and large values. Large values are further penalized by Geman-McClure error function

Elastic loss: $L_{\text {elastic }}(\mathbf{x})=\|\log \boldsymbol{\Sigma}-\log \mathbf{I}\|_F^2=\|\log \boldsymbol{\Sigma}\|_F^2$:
- For a fixed latent code $\omega_i$, deformation field $T$ is non-linear mapping from observed cordinates ($\mathbb{R}^3$) to canonical coordinates ($\mathbb{R^3}$).
Jacobian $J_T(x)$ of the mapping ($T$) at a point $x \in \mathbb{R^3}$ describes **the best linear approximation of the transformation at that point**. 
- Because of the continuous formulation of the surface (instead of discrete) we can directly compute $J_T$ through differentiation of the MLP.
- How to penalize deviation from the $\mathbf J_T$ jacobian?:
  - singular-value decomposition $\mathbf{J_T = U\Sigma V^T}$
  - penalize the deviation from the closest rotation as $\lvert\lvert \mathbf{J_T - R}\rvert\rvert^2_F$ where $\mathbf{R = VU^T}$
  - $\lvert\lvert\cdot\rvert\rvert_F$ is a [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) defined as $\sqrt{\sum_{i=1}^m \sum_{j=1}^n\left|a_{i j}\right|^2}$ (sqrt of sum of squared elements)
- the log of singular values $\log\Sigma$ gives equal weight to contraction and expansion of the same factor (it performs better) 
- because of that, $\log \mathbf{I}$ becomes $\mathbf{0}$

Robust loss $L_{\text {elastic-r }}(\mathbf{x})=\rho\left(\|\log \boldsymbol{\Sigma}\|_F, c\right)$:
- $\rho(x, c)=\frac{2(x / c)^2}{(x / c)^2+4}$ - Geman-McClure error function with $c = 0.03$
- this functino causes the gradients of loss to fall off to zero for large values of the argument (x)
- therefore, reducing the influence of outliers during the training
- ![](imgs/nerfies-geman-mclure.jpg)

Weighting:
- deformation field behaves freely in empty space since subject moving relative to the background requires non-rigid deformation somewhere in space
- elastic penality is weighted at each sample along the ray by its contribution to the rendered view ($w_i$)
- equation 5 from the original NeRF paper: $w_i = T_i(1-\exp(-\sigma_i\delta_i))$
  - $T(t)$ is accumulated transmittance along the ray from $t_n$ to $t$ (the probability that ray ravels from $t_n$ to $t$ without hitting any other particle). The higher the value, the probability of "something being there" is smaller. 
  - $\sigma_i$, how "dense" is the particle that is being hit? If the value is high, the particle is dense and the strength of the ray will be lower (next particle will recieve less 'energy' from the ray).

## 3.4 Background regularization
Background regularization: $L_{\mathrm{bg}}=\frac{1}{K} \sum_{k=1}^K\left\|T\left(\mathbf{x}_k\right)-\mathbf{x}_k\right\|_2$

Important: $x_k$ are static 3D points

Takeaway: Prevents background from moving. Given a set of 3D points in the scene which we know should be static, we can penalize any deformations at these points. It also aligns observation frame to canonical frame.


## 3.5 Coarse-to-fine deformation regularization
Takeaway: introduce parameter $\alpha$ that windows the fequency bands and use it in weight $w_j$. For each $\alpha$ increase new and higher frequencies are *unlocked* because their weights are no longer 0.

$m$ maximum number of bands:
- small values: low resolution, too smooth
- large values: high resolution
- ![](imgs/nerfies-m-value-change.jpg)
- high $m=8$ caputres smile change well but head turn is awful
- low $m=4$ has worse smile but better head turn

Positional encoding can be interpreted in terms of Neural Tangent Kernel (NTK) of NeRF's MLP:
- statoionary interpolating kernel
- $m$ controls a tunable "bandwidth" of the kernel
- small number of frequencies includes a wide kernel => under-fitting of the data
- large number of frequencies includes a narrow kernel => over-fitting

Solution: introduce parameter $\alpha$ that windows the fequency bands and use it in weight $w_j$:

$\omega_j(\alpha) = \frac{1-cos(\pi\text{clamp}(\alpha - j, 0, 1))}{2}$:
- $j$ each frequency band
- $\alpha \in [0, m]$ is being lineary annealed
  - sliding a truncated Hann window (where the left side is clamped to 1 and the right side is clamped to 0)
  - ![](imgs/nerfies-hann-function.png)
  - ![](imgs/nerfies-sliding-window.jpg)
  - $\alpha=1$
    - for frequency~~0 weight is 1 (very low frequency)
    - for frequency=0.5 weight is 0.5
    - for frequency=1 weight is 0
  - $\alpha=3$
    - weights for all previous frequencies are 1!
    - we start by including the *new* frequencies ([2, 3]) 
    - for frequency 2.5 weight is 0.5
    - for frequency 3 weight is still ~0

Positional encoding $\gamma_\alpha(x) = \left(\mathbf{x}, \cdots, w_k(\alpha) \sin \left(2^k \pi \mathbf{x}\right), w_k(\alpha) \cos \left(2^k \pi \mathbf{x}\right), \cdots\right)$
- during training $\alpha(t) = \frac{mt}{N}$
- $t$ current training iteration
- $N$ number of training itterations until $\alpha$ reaches maximum number of frequencies $m$

## Notes

Non-rigid shape: shapes which change position/rotation but also position of individual parts of the object. The problem is how to map the previous point to the new one.
![](./imgs/nerfies-non-rigid.jpg)
Jacobian - matrix of gradients (partial derivatives)

![](imgs/jacobian.jpg)

# Mip-NeRF 360: Unbounded
https://jonbarron.info/mipnerf360/

## 1. Introduction

Problems with NERF:

- works well on fixed distance but blurred in close-up views and contain artifacts in distant views
- potential solution: supersampling each pixel - march multiple rays through the pixel (very expensive!)
  - each ray takes several hours

Mip:
- pre-filtering: mipmap represents a signal (image) - at a set of different discrete downsamping scales - and selects appropriate scale
  - computation is shifted from render time (anti-aliasing) to precompute phase (mipmap). Created only once for a given texture regardless of number of times it needs to be rendered
- Input: 3d Gausian that representes the region over which the radiance field should be integrated
- Rendering: querying mip-NeRF at intervals along a cone and using Gaussians which approximate the conical frustums corresponding to the pixel.
Integradted positional encoding: generalization of NeRF's positional encoding (allows a region of space to be featurized)
  - it encodes 3d position and its surrounding Gaussian region

## 2. Related work

## 5 Conclusion

## Notes
- trains single NN that models the scene at multiple scales
- cats cones and encodes positions and sizes of conical frustums
- extends NeRF to represent the scene at a continuously-value scale
- rendering anti-aliased conical frustums instead of rays
- preserves fine details
- 7% faster than NeRF
- 1/2 size of NeRF
- reduces avg. error by 17%
- reduces avg. error by 60% on a challenging multiscale variant dataset

reference:

frustum - portion of a solid (normally a pyramid or a cone) that lies between one or two parallel planes cutting it

mipmap

Todo:
- box downsampling

## Technical

run `sudo rm /etc/apt/sources.list.d/cuda*` after cuda uninstalling

`export TCNN_CUDA_ARCHITECTURES=75` <https://developer.nvidia.com/cuda-gpus>

when building colmap fix flags: <https://github.com/colmap/colmap/issues/1418#issuecomment-1111406726>

Use ceres-solver 2.1.0 when building

Can't find cuda? Set proper CUDA_HOME, CUDA_PATH and add cuda/bin to path

# Directory structure

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
