
# NeRF (Neural Radiance Fields) research

## Log

2022-11-17:

- note taking and complete understanding of Mip-NeRF paper https://github.com/google/mipnerf
- setup Docker image which installs everything needed to run the [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) project
  - also because my local machine has <= 6GB VRAM and I'm forced to use the server
- took pictures and videos of my student id card in multiple conditions
  - perfect lightning
  - lower resolution image
  - bad lightning conditions
  - half blurry images
  - video of the id card
- plan: try out all dataset variations on models out of the box models: [instant-ngp](https://docs.nerf.studio/en/latest/nerfology/methods/instant_ngp.html), [mip-nerf](https://docs.nerf.studio/en/latest/nerfology/methods/mipnerf.html), [nerfacto](https://docs.nerf.studio/en/latest/nerfology/methods/nerfacto.html)

2022-11-03:

- note taking and complete understanding of Nerfies paper https://nerfies.github.io/
- found "nerfacc" which offers efficient ray marching and volumetric rendering https://github.com/KAIR-BAIR/nerfacc
- Light Field Neural Rendering: great for hologram reflection https://light-field-neural-rendering.github.io/
- plan: further reserach and collecting more approaches to try and combine them:
  - plenoxels (no neural networks) https://alexyu.net/plenoxels/
  - Mip-NeRF 360 (CVPR 2022) https://jonbarron.info/mipnerf360/
  - Direct Voxel Grid Optimization (CVPR 2022) https://sunset1995.github.io/dvgo/

2022-10-27:

- caculator results (out-of-the-box [nerfacto](https://docs.nerf.studio/en/latest/nerfology/methods/nerfacto.html) model):
  - ![](./imgs/first-caculator.jpg)
  - shot 10 images of a caculator with variying extreme angle differences https://www.youtube.com/watch?v=0SqYMH9wiwg
  - used nerfstudio to create a first result https://github.com/nerfstudio-project/nerfstudio
  - model "Nerfacto". "Flagship method which uses techniques from Mip-NeRF 360 and Instant-NGP." https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/nerfacto.py
  - tried to perform training with instant-ngp (instant nerf) but ran out of memory, will try it on a dedicated server
- installing and preparing starting https://github.com/nerfstudio-project/nerfstudio
- reading Nerfies: Deformable Neural Radiance Fields
- reading Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields

## To-do

- [ ] .keep

## Nerfies: Deformable Neural Radiance Fields

https://nerfies.github.io/

- deformation field: wrap each observed point into deformation field (MLP) which is also optimized
- coarse-to-fine optimization: start by zeroing out the higher frequency at the start of the optimization (low level details) and the network will learn smooth deformations. Later, introduce higher frequencies
- two phones: one phone can be used for training while the other can generate test data

### 1. Introduction

"Deformation field is conditioned on a per-image learned latent code" => this suggests that the deformation field varies between each observation

### 3. Deformable Neural Radiance Fields

![](imgs/nerfies-arh.jpg)

- each image has its own latent deformation code ($\omega$) and latent appearance code ($\psi$)
- NeRF isn't querying from the observation frame (classical approach, ($x, y, z$) ), rather, it's querying from the canoncial frame ($x', y', z'$)
- deformation field is key extension that allows representation of moving subjects
- elastic regularization (3.3), background regularization (3.4), coarse-to-fine (3.5), has to be introduced to avoid under-constrained optimization of deformation field + NeRF

### 3.1 NeRF -- memory refresh

- NeRF is a function $F : ({\mathbf x,d,\psi_i}) \rightarrow ({\mathbf c}, \sigma)$ which maps:
  - 3D position ${\mathbf x} = (x,y,z)$ and
  - viewing direction $\mathbf{d} = (\phi, \theta)$
  - to $\mathbf{c} = (r,g,b)$ and
- In pratice, NeRF maps $\mathbf x$ and $\mathbf d$ using a sinusodial positional encoding $\gamma : \mathbb{R^3} \rightarrow \mathbb{R^{3+6m}}$ where $m$ is hyperparameter, total number of frequency bands
- $\gamma(\mathbf{x})=\left(\mathbf{x}, \cdots, \sin \left(2^k \pi \mathbf{x}\right), \cos \left(2^k \pi \mathbf{x}\right), \cdots\right)$

$\psi_i$ latent appearance code exists for each image and it **modulates the color output** to handle interpolation between exposure/white balanace between frames

### 3.2 Neural Deformation Fields

**Takeaway**: create a mapping which maps original points to a canoncial template using MLP and use rigid rotation (optimizing only one rotation instead of rotation for each point)

How to allow NeRF to reconstruct non-rigidly deformations?

- Use canonical template of the scene
- To create canonical template they define a mapping $T_i : (x, \omega_i) \rightarrow x'$
- $\omega_i$ is a per-frame learned latetnt deformation code

Problem: rotating a group of points with a translation field requires a different translation for each point.

Solution: use proper rigid transformations in a 3-dimensional Euclidean space, denoted **SE(3)**.

A mapping $W : (x, ωi ) \rightarrow SE(3)$ encodes rigid motion which allows rotation of a set of points which are far away from one another. SE(3) requires only **one rotation parameter** while translation field requires parameter for each point. It's easier to optimize one parameter.

<details><summary>
SE(n): proper rigid transformations
</summary>
https://www.wikiwand.com/en/Rigid_transformation

rigid transformation (also called Euclidean transformation or Euclidean isometry) is a geometric transformation of a Euclidean space that preserves the Euclidean distance between every pair of points. Any object will keep the same shape and size after a proper rigid transformation.

Transformations that also perserve handedness of objects (orientability) are called proper rigid transformation.

The set of proper rigid transformations is called special Euclidean group, denoted SE(n).
</details>

<details><summary>
SE(3) paper snippet
</summary>

![](imgs/nerfies-se3.jpg)
</details>

### 3.3 Elastic regularization

**Takeaway**: create jacobian and prenalize the deviation of signular values of $J$ from $1$ (closest rotation (?)) and prenalize large values. Large values are further penalized by Geman-McClure error function.

Problem: object is moving backwards and is visually shrinking
Solution: use elastic energy to measure deviation of local deformations from a rigid motion

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

### 3.4 Background regularization

**Takeaway**: Prevents background from moving. Given a set of 3D points in the scene which we know should be static, we can penalize any deformations at these points. It also aligns observation frame to canonical frame.

Background regularization: $L_{\mathrm{bg}}=\frac{1}{K} \sum_{k=1}^K\left\|T\left(\mathbf{x}_k\right)-\mathbf{x}_k\right\|_2$

Important: $x_k$ are static 3D points

### 3.5 Coarse-to-fine deformation regularization

**Takeaway**: introduce parameter $\alpha$ that windows the fequency bands and use it in weight $w_j$. For each $\alpha$ increase new and higher frequencies are *unlocked* because their weights are no longer 0.

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

### Notes

Non-rigid shape: shapes which change position/rotation but also position of individual parts of the object. The problem is how to map the previous point to the new one.
![](./imgs/nerfies-non-rigid.jpg)
Jacobian - matrix of gradients (partial derivatives)

![](imgs/jacobian.jpg)

## Mip-NeRF

https://github.com/google/mipnerf

**Takeaway**: render anti-aliased conical frustums instead of rays which reduce artifacts and improve fine details. 7% faster than NeRF, 0.5 * NeRF size, reduced error from 17% to 60%.

### 1. Introduction

**Problems with NeRF**: Original NeRF is slow, has blurred close-up and contains artifacts in distant views. Can we supersample each pixel by marching multiple rays through it? No, it's very expensive.

**Takeaway**:  The input to the mip-NeRF is 3D Gaussian that represents the region over which the radiance field should be integrated. Mip-NeRF’s scale-aware structure allows to merge "coarse" and "fine" MLP.

integrated positional encoding (IPE):

- encodes 3D positions **and its surrounding Gaussian region**
- encode a **region of space** instead of a single point in space

### 2 Related work

Anti-aliasing in rendering: supersampling or pre-filtering:

- supersampling: cast multiple rays per pixel while rendering (sample closer to Nyquist frequency). Very impractical
- prefiltering: low-pass-filtered version of the scene => decrease Nyquist frequency required to render the scene without aliasing. This can be precomputed ahead of time
  - prefiltering can be thought of as tracing a cone instead of a ray through each pixel
  - precomputed multiscale representation of the scene content (sparse voexel octree or a mipmap)

Mip-NeRF related notes:

- mutliscale representation cannot be precomupted because the **scene's geometry is not known ahead of time**
- Mip-NeRF must learn prefiltered representation of the scene during training
- scale is continuous instead aof discrete

Scene representation for View Syntehsis:

- mesh-based representations:
  - pros: can be stored efficiently, are compatible with graphics rendering pipelines
  - cons: gradiant-based methods to optimize mesh geometry are difficutl due to discontinuities and local minima
- volumetric representations: better
  - gradient-based learning to train NN to predict voxel grid (3d cube made up of unit cubes) representation of scenes

coordinate-based neural representations:

- replacing discrete representations (3D Scenes) with MLP (NeRF)

### 3 Method

**Takeaway**: Instead of performing point-sampling along each ray, we divide the cone being cast into a series of conical frustums (cones cut perpendicular to their axis) and integrated positional encoding (IPE) instead of PE. Now the MLP can reason about the size and shape of each conical frustum instead of just its centroid. Because of IPE, "coarse" and "fine" are merged into single MLP (speed and size are improved by 50%).

### 3.1 Cone tracing and positional encoding

**Takeaway**: Approximate the conical frustum with a multivariate Gaussian. Parameters ($\mu, \sigma$) can be found in closed form and are. IPE is expectation of a positionally-encoded coordinate distributed according to the Gaussian. Diagonal of $\Sigma$ is needed which is cheap. IPE is roughly as expensive as PE to construct. Hyperparameter $L$ (positional encoding) is not needed anymore.

![](imgs/mip_main_fig.jpg)

- Images are rendered one pixel at the time
- Apex (starting point) of the cone lies at $o$ (eye, observation point) and the radis of the cone $o + d$ (the further you go the radius gets bigger) parameterized as $\dot{r}$
- $\dot{r}$ is width of the pixel scaled by $\frac{2}{\sqrt{12}}$
- this yields a cone whose selection on the image plane has variance in x and y that maches the variance of the pixel's footprint
- set of positions $\mathbf{x}$ that lie within conical frustum between two $t$ values $\in [t_0, t_1]$
- featurized representation: expected positional encoding of all coordinates that lie withing the conical frustum: $\gamma^*\left(\mathbf{o}, \mathbf{d}, \dot{r}, t_0, t_1\right)=\frac{\int \gamma(\mathbf{x}) \mathrm{F}\left(\mathbf{x}, \mathbf{o}, \mathbf{d}, \dot{r}, t_0, t_1\right) d \mathbf{x}}{\int \mathrm{F}\left(\mathbf{x}, \mathbf{o}, \mathbf{d}, \dot{r}, t_0, t_1\right) d \mathbf{x}}$
  - how is this featured computed efficiently? (integral in the numerator has no closed form)
  - approximate the conical frustum with multivariate Gaussian => IPE
  - compute mean and covariance of $F(x, \cdot)$
  - Gaussian can be fully characterized by 3 values:
    - mean distance along the ray $\mu_t$
    - the variance along the ray $\sigma^2_t$
    - variance perpendicular to ray $sigma^2_r$
  - quantitues are parameterized with:
    - middle point $t_u = \frac{(t_o + t_1)}{2}$
    - half-width $t_\delta = \frac{(t_1 - t_0)}{2}$
    - they are critical for numerical stability
  - Gaussian from the coordinate frame of conical frustum --into--> world coordinates:
    - $\mathbf{\mu} = \mathbf{o} + \mu_t\mathbf{d}$
    - $\mathbf{\Sigma} = \sigma^2_t(\mathbf{dd}^T)+\sigma^2_r(\mathbf{I - \frac{dd^T}{||d||^2_2}})$
  - IPE is derived via closed form !
  - it relies on the marginal distribution of $\gamma(x)$ and diagonal covariance matrix $\Sigma_\gamma$ which depends on the diagonal of the 3D position's covariance $\Sigma$
  - IPE is roughly as expensive as PE
  - if period is smaller than interval (PE over that interval will oscillate repeatedly) then encoding that encoding is sacled towards zero
  - IPE perserves frequencies that are constant over an interval and softly removes frequencies that vary over an interval
  - PE perserves all frequencies up to hyperparameter $L$
  - IPE remove hyperparameter $L$ (set it to large value and don't tune it)
  - ![](imgs/mip_fig4.jpg)

### 3.2 Architecture

**Takeaway**: mip-NeRF works for single scale. One parameter $\Theta$ instead of two (classic NeRF). The loss function still includes "coarse" and "fine" losses.

- cast cone instead of ray
- instead of sampling $n$ values for $t_k$ they sample $n+1$ values
- features are passed into the MLP to produce density $\tau_k$ and color $c_k$
- IPE encodes scale (no "coarse" and "fine") so MLP has only parameters $\Theta$, model is cut in half and renderings are more accurate
- optimization problem: $\min _{\Theta} \sum_{\mathbf{r} \in \mathcal{R}}\left(\lambda\left\|\mathbf{C}^*(\mathbf{r})-\mathbf{C}\left(\mathbf{r} ; \Theta, \mathbf{t}^c\right)\right\|_2^2+\left\|\mathbf{C}^*(\mathbf{r})-\mathbf{C}\left(\mathbf{r} ; \Theta, \mathbf{t}^f\right)\right\|_2^2\right)$
- "coarse" loss is balanced against "fine" loss by setting $\lambda = 0.1$
- coarse samples $t^c$: produced with stratified sampling
- fine samples $t^f$: sampled from resulting alpha compositing weights $w$ using inverse transform sampling
- mip-nerf samples 128 coarse and 128 fine
- weights $w$ for $t^f$ are modified $w_k^{\prime}=\frac{1}{2}\left(\max \left(w_{k-1}, w_k\right)+\max \left(w_k, w_{k+1}\right)\right)+\alpha$
- $w$ is filtered with 2-tap max filter, 2-tap blur filter
- $\alpha = 0.01$ is added before it is re-normalized to sum of 1

### 4. Results

PSNR, SSIM, LPIPS

Mip-NeRF reduces average error by 60% on this task
and outperforms NeRF by a large margin on all metrics
and all scales.

### Notes

mip-NeRF hyperparameters:

1. number of samples $N$ drawn at each of two levels (N = 128)
2. histogram padding $\alpha$ on the coarse trainsmittance weights ($\alpha = 0.01$). Larger $\alpha$ baisases the final samples toward uniform distribution
3. multiplier $\lambda$ on the "coarse" component of the loss function ($\lambda = 0.1$)

Three more hyperparameters from NeRF are excluded:

1. Number of samples $N_c$ drawn for "coarse" MLP
2. Number of samples $N_f$ drawn for "fine" MLP
3. Degree $L$ used for spatital positional encoding ($L = 10$)

Activations:

- softplus $log(1+exp(x-1))$ instead of ReLU
- this is becase MLP sometimes emites negative values everywhere (gradients from $\tau$ are then zero and optimization fails)
- shift by -1 is equivalent to initializing biases and produce $\tau$ to $-1$ and casues intial $\tau$ values to be small
  - faster optimization in the beginning of the training as dense scene content causes gradients from scence content "behind" that dense content to be suppressed (front scene has the 'edge')
- widened sigmoid instead of sigmoid to produce color $c$
  - avoid saturation in tails of the sigmoid (black and white pixels)

Other:

- trains single NN that models the scene at multiple scales
- cats cones and encodes positions and sizes of conical frustums
- extends NeRF to represent the scene at a continuously-value scale
- rendering anti-aliased conical frustums instead of rays
- preserves fine details
- 7% faster than NeRF
- 1/2 size of NeRF
- reduces avg. error by 17%
- reduces avg. error by 60% on a challenging multiscale variant dataset
- frustum - portion of a solid (normally a pyramid or a cone) that lies between one or two parallel planes cutting it

## (work in progress) instant-ngp

.keep

## (work in progress) Mip-NeRF 360: Unbounded

.keep

## Resources

https://dellaert.github.io/NeRF22/ list of nerfs

## Technical

- run `sudo rm /etc/apt/sources.list.d/cuda*` after cuda uninstalling
- `export TCNN_CUDA_ARCHITECTURES=75` https://developer.nvidia.com/cuda-gpus
- when building colmap fix flags: https://github.com/colmap/colmap/issues/1418#issuecomment-1111406726
- Use ceres-solver 2.1.0 when building
- Can't find cuda? Set proper CUDA_HOME, CUDA_PATH and add cuda/bin to path

## Commands

Turn on viewer:

ns-train nerfacto --vis viewer --viewer.websocket-port=7010 --data data/student-id/light2/ --trainer.load-dir outputs/light2/nerfacto/2022-12-11_183713/nerfstudio_models --viewer.start-train False

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
