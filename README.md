# Variational Auto-Encoder (VAE) Tensorflow
An implementation of variational auto-encoder (VAE) for MNIST and FreyFace descripbed in the paper: [Auto-Encoding Variational Bayes, ICLR2014](https://arxiv.org/pdf/1312.6114.pdf) by Kingma et al.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/48118604-176de680-e2b0-11e8-9e9e-c4be981465f1.png" width=700)
</p>  
  
## Requirements
- tensorflow 1.10.0
- python 3.5.5
- numpy 1.14.2
- matplotlib 2.2.2
- scipy 0.19.1
- pillow 5.0.0

## Applied VAE Structure
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48119259-32415a80-e2b2-11e8-877f-1b264c2ccd84.png" width=800>
</p>

## Results
### 1. MNIST Reconstruction
- 2-D latent space  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48129147-fb783e00-e2cb-11e8-856c-22b204afa87a.png" width=600>
</p>

- 5-D latent space  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48129193-177bdf80-e2cc-11e8-891e-49e9aa9235fb.png" width=600>
</p>

- 10-D latent space  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48129213-22367480-e2cc-11e8-86a3-275e243858a1.png" width=600>
</p>

- 20-D latent space  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48129222-2bbfdc80-e2cc-11e8-93f2-3927762d4500.png" width=600>
</p>

### 2. Frey Face Reconstruction
- 2-D latent space  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48129548-2e6f0180-e2cd-11e8-89f7-69cf96ad9745.png" width=400>
</p>

- 5-D latent space  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48129583-4c3c6680-e2cd-11e8-9b37-963ae8612caa.png" width=400>
</p>

- 10-D latent space  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48129594-552d3800-e2cd-11e8-9169-35e0991e1fc0.png" width=400>
</p>

- 20-D latent space  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48129665-8a398a80-e2cd-11e8-8d8e-b53d2f722a89.png" width=400>
</p>

### 3. MNIST Denoising
- 2-D latent space 
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48130334-ac340c80-e2cf-11e8-9be0-44b6b8a676cb.png" width=700>
</p>

- 5-D latent space  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48130341-b48c4780-e2cf-11e8-85f9-18afa4b722c4.png" width=700>
</p>

- 10-D latent space  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48130352-be15af80-e2cf-11e8-8c47-8fb4fd09aad6.png" width=700>
</p>

- 20-D latent space  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48130364-c837ae00-e2cf-11e8-80fe-a65edf855ebe.png" width=700>
</p>

### 4. Learned MNIST Manifold and Distribution of Labeled Data
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48130706-dafeb280-e2d0-11e8-9ef4-fe13feb297ad.png" width=900>
</p>

### 5. Learned Frey Face Manifold
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48130864-6710da00-e2d1-11e8-83c1-8de5e6321da9.png" width=500>
</p>

## Documentation
### Download Dataset
Download 'FreyFace' dataset from the [link](https://cs.nyu.edu/~roweis/data.html) that mentioned in the VAE paper. 'MNIST' dataset can be downloaded automatically from the code.

### Directory Hierarchy
``` 
.
│   VAE
│   ├── src
│   │   ├── dataset.py
│   │   ├── vae.py
│   │   ├── main.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   └── utils.py
│   Data
│   ├── mnist
│   └── freyface
```  
**src**: source codes of the VAE

### Training VAE
Use `main.py` to train a VAE network. Example usage:

```
python main.py
```
 - `gpu_index`: gpu index, default: `0`
 - `batch_size`: batch size for one feed forward, default: `128`
 - `dataset`: dataset name from [mnist, freyface], default: `mnist`
 
 - `is_train`: training or inference mode, default: `True`
 - `add_noise`: boolean for adding salt & pepper nooise to input image, default: `False`
 - `learning_rate`: initial learning rate for Adam, default: `0.001`
- `z_dim`: dimension of z vector, default: `20`

 - `iters`: number of interations, default: `20000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `5000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_batch`: number of sampling images for check generator quality, default: `100`
 - `load_model`: folder of save model that you wish to test, (e.g. 20181107-2106_False_20). default: `None` 
 
### Test VAE
Use `main.py` to test a VAE network. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20181107-2106_False_20
```
Please refer to the above arguments.

### Citation
```
  @misc{chengbinjin2018vae,
    author = {Cheng-Bin Jin},
    title = {VAE-Tensorflow},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/VAE-Tensorflow},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project refered some code from [hwalsuklee](https://github.com/hwalsuklee/tensorflow-mnist-VAE).  
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
