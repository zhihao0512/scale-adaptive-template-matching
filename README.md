# scale-adaptive-template-matching
A MATLAB/C++ version code of the paper "Scale-Adaptive NN-based Similarity for Robust Template Matching", IEEE transactions on instrumentation and measurement, 2020.
## Installation
We tested using MATLAB 2018a and VS2017 64bit.

To compile mex functions:
```
--mex DIS_scan.cpp
--mex WSDIS_scan.cpp
```
Download imagenet-vgg-verydeep-19.mat to "./utils/deepFeatures".
## Running
Run DEMOrun.m
## References
I. Talmi, R. Mechrez, and L. Zelnik-Manor. "Template matching with deformable diversity similarity" In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 175-183. 2017. url: https://github.com/roimehrez/DDIS

Dekel, Tali, Shaul Oron, Michael Rubinstein, Shai Avidan, and William T. Freeman. "Best-buddies similarity for robust template matching." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2021-2029. 2015. Url: http://people.csail.mit.edu/talidekel/Best-Buddies%20Similarity.html

Olonetsky, Igor, and Shai Avidan. "Treecann-kd tree coherence approximate nearest neighbor algorithm." In European Conference on Computer Vision, pp. 602-615. Springer Berlin Heidelberg, 2012. url: https://github.com/uva-graphics/patchtable/tree/master/patchtable/TreeCANN

Muja, Marius, and David G. Lowe. "Fast approximate nearest neighbors with automatic algorithm configuration." VISAPP (1) 2, no. 331-340 (2009): 2. url: http://www.cs.ubc.ca/research/flann/

MatConvNet: CNNs for MATLAB. url: http://www.vlfeat.org/matconvnet/
