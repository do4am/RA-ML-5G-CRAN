# Internship topic: Application of Machine learning to resource allocation for CRAN based 5G system
Imtiaz, S., Koudouridis, G.P., Ghauch, H. and Gross, J., 2018. Random forests for resource allocation in 5G cloud radio access networks based on position information. EURASIP Journal on Wireless Communications and Networking, 2018(1), p.142.

# Idea
CRAN is Cloud Radio Access Network
<p align="center">
  <img src="https://user-images.githubusercontent.com/15823161/52285347-73571280-2966-11e9-94d8-fec46d2230a7.png">
</p>
The resource allocation (RA - a combination of Transmission Beam Tx, Received Filter Rx, and MCS M) is done via the downlink communication between RRH and UE. The scenario of this project is a Remote Radio Head (RRH) and 1 User(UE) (the red region). <br />

The traditional approach is to do RA based on Channel State Information (CSI) has several significant drawbacks: Computational cost and overhead time-slots in a TDD system. A learning-based approach is designed to overcome the problem. <br />

The idea is to train a network that can do the RA based on user location only. <br />

This method is possible because CRAN is a dense network, which means there will be a lot more line-of-sight (LOS) links between UE and base-station (RRH). This makes the behavior of the channel more deterministic, and thus, the channel impulse response at one position can be simulated. <br />

To generate the dataset for training: Exhaustive search is used as follows <br />
First, all possible channel impulse responses matrices are simulated and stored in H. <br />
Given a street map. Randomly select a position on the map (euclidean-based), embedded this position with all channel matrices in H. Run exhaustive search to find which RAs yield the best goodput for each channel matrix. Collect them. Repeat the step with another random position until we find the amount of data is sufficient for training. <br />

Dataset: Input: position (euclidean-based); Labels/Classes: RAs corresponding to that position. <br />

This dataset has a problem: Overlapping as shown in the picture.
<p align="center">
  <img src="https://user-images.githubusercontent.com/15823161/52287627-d21e8b00-296a-11e9-9e8b-52aa4f4cb889.png">
</p>
Thesis Topic: Data Processing Techniques to Optimise the learning based resrouce allocation in wireless communication systems. 
Visualization with Scatter Plot of Raw dataset:
<p align="center">
  <img src="https://user-images.githubusercontent.com/15823161/59569712-b43a0300-908d-11e9-83ae-09a3b7f6267d.jpg" width="300" > 
</p>
As can be seen from the Scatter Plot, each of the colors represents for a class/ resource allocation. The scatter plot of all the samples in a street section of 6m x 25m, and the distribution of the classes among those samples. Since one sample may belong to multiple classes, there is an overlapping issue between types. <br />

The original dataset has the problem of overlapping - one sample belongs to more than one class. The data processing algorithms, called Merger and Nearest Neightbors (NN) are applied to remove all the overlapping issue, make sure one sample belongs to a single class.

Processing directly to the raw dataset before feeding them through a neural network.
After being processed:
<p align="center">
  <img src="https://user-images.githubusercontent.com/15823161/59569723-d6cc1c00-908d-11e9-9e8a-7a9478101cb6.jpg" width="300"> 
</p>

As can be seen clearly, the processed picture looks somehow cleaner, the classes become more distinct. If you are interested in the insightful analysis of the result, please feel free to send me an email (Given at the end of this note).

Overall system performance gained 10% without losing a single bit of raw data. 

# Built with 
Python 2.7 <br />
Matlab

Due to copyright issue the detail of the algorithms cannot be published. Please contact me via mhndo@kth.se if you want to discuss on something.
