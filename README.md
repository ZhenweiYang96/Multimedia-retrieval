# Content-based 3D Shape Retrieval System
Authors: Zhenwei Yang, Oscar Hsieh

## Description:
The multimedia retrieval system is able to yield similar shapes from a 3D database if the user inputs a specific object. 

The database used to develop the retrieval system is the Labelled PSB Dataset, consisting of 380 3D shapes belonging to 19 categories. The 3D objects in the database are stored in Object File Format (OFF) files. Each object has its own ID number ranging from 1 to 260 and 281 to 400. All the shapes were converted to a triangle mesh in the backend.

## Noramlization
The 3D shapes in the database should be firstly normalized to faciliate the feature extraction. The noramlization includes 5 steps:
* Remeshing: subsampling the faces of the polymesh to reduce the face count；
* Translation: translate the barycenter to the origin of the coordinate system；
* Alignment: ensure every mesh to present the uniform pose；
* Flipping: the most mass is always to the left part (negative half-space);
* Scaling: ensure each mesh is scaled to fit a unit-size cube.

The noramlization step is visualized as below:

<img src="https://raw.githubusercontent.com/ZhenweiYang96/Multimedia-retrieval/master/Image/normalization.png" width="600" height="300"/>

## Feature Extraction
There were 5 elementary features:
* Surface area: the summation of all individual triangle surfaces;
* Sphericity: detect how close a shape is to a sphere;
* AABB volume: The volume of the axis-aligned bounding box;
* Diameter: the largest distance between two vertices of a shape;
* Eccentricity: the ratio of largest to smallest eigenvalues of covariance matrix

and 5 distributional features:
* A3: the angle (&theta;) between 3 random vertices;
* D1: the distance between barycentre and random vertex;
* D2: the distance between 2 random vertices;
* D3: the square root of area of triangle given by 3 random vertices;
* D4: the cube root of volume of tetrahedron formed by 4 random vertices

## GUI
The demo of the interface is shown as below:

<img src="https://raw.githubusercontent.com/ZhenweiYang96/Multimedia-retrieval/master/Image/interface.jpg" width="600" height="500"/>

## Matching System
Two matching system provided: customized weight matching system and scalability techinque. The former one is based on Earth Mover's Distance (EMD) and Euclidean Distance. The later one is based on the nearest neighbour engine. The engine combines k nearest neighbours and fixed-raduis nearest neighbours. Thus the number of output shapes is not fixed.

One sample output of the matching system is:

<img src="https://raw.githubusercontent.com/ZhenweiYang96/Multimedia-retrieval/master/Image/example_matching.png" width="600" height="300"/>

One sample output of the scalability technique is:

<img src="https://raw.githubusercontent.com/ZhenweiYang96/Multimedia-retrieval/master/Image/example_scalability.png" width="600" height="200"/>

## Evaluation
The users are also allowed to do the evaluation based on the confusion matrix.
