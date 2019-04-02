# phomgem
Persistent Homology for Generative Models

phomgem is a Python and R library that proposes to evaluate the quality of the generated adversarial samples using persistent homology. For some real-world applications, different than computer vision, we cannot assess visually the quality of the generated adversarial samples. Therefore, we have to use other metrics. Here, we rely on persistent homology because it is capable to acknowledge the shape of the data points, by opposition to traditional distance measures such the Euclidean distance.

The generative models are trained with Python to produce adversarial samples savec in csv files.

The persistent homology features and the bottleneck distance are evaluated with the TDA package of R. 

<p float="center">
  <img src="https://github.com/dagrate/phomgem/blob/master/images/barcodes_originSamples.png" width="300"/>
  <img src="https://github.com/dagrate/phomgem/blob/master/images/barcodes_originSamples.png" width="300"/>
</p>

<figure>
  <img src="https://github.com/dagrate/phomgem/blob/master/images/barcodes_originSamples.png" width="400"/>
  <figcaption>Persistent Diagram</figcaption>
  <img src="https://github.com/dagrate/phomgem/blob/master/images/barcodes_originSamples.png" width="400"/>
  <figcaption>Barcodes</figcaption>
</figure>


----------------------------

## Dependencies

The library uses **Python 3** and **R** with the following modules:
- numpy (Python 3)
- scipy (Python 3)
- matplotlib (Python 3)
- pandas (Python 3)
- pylab (Python 3)
- sklearn (Python 3)
- keras (Python 3)
- functools (Python 3)
- TDA (R)
- TDAmapper (R) -> only if you want to play with the mapper algorithm

It is advised to install BLAS/LAPACK to increase the efficiency of the computations:  
sudo apt-get install libblas-dev liblapack-dev gfortran

----------------------------

## Citing

If you use the repository, please cite:

```bibtex
@inproceedings{charlier2019pho,
  title={PHom-GeM: Persistent Homology for Generative Models},
  author={Charlier, Jeremy and State, Radu and others},
  booktitle={The 6th Swiss Conference on Data Science (SDS), 2019 IEEE International Conference},
  pages={},
  year={2019},
  organization={IEEE}
}
```
