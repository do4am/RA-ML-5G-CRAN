# Internship topic: Application of Machine learning to resource allocation for CRAN based 5G system
Imtiaz, S., Koudouridis, G.P., Ghauch, H. and Gross, J., 2018. Random forests for resource allocation in 5G cloud radio access networks based on position information. EURASIP Journal on Wireless Communications and Networking, 2018(1), p.142.

# Idea
CRAN is Cloud Radio Access Network

![image](https://user-images.githubusercontent.com/15823161/52285347-73571280-2966-11e9-94d8-fec46d2230a7.png)

The resource allocation (RA - a combination of Transmission Beam Tx, Received Filter Rx and MCS M) is done via the downlink communication between RRH and UE. The scenario of this project is 1 Remote Radio Head (RRH) and 1 User(UE) (the red region). <br />

Traditional approach is to do RA based on Channel State Information (CSI) has several signigficant drawbacks: Computational cost and overhead time-slots in TDD system. A learning-based approach is designed to overcome the problem. <br />

The idea is to train a network that can do the RA based on user location only. <br />

This method is possible because CRAN is a densed network which means there will be a lot more line-of-sight (LOS) links between UE and base-station (RRH). This makes the behavior of the channel become more deterministic and thus the channel impulse response at one position can be simulated. <br />

To generate the dataset for training: Exhaustive search is used as follows <br />
First all possible channel impulse resonses matrices are simulated and stored in H. <br />
Given a street map. Randomly select a position on the map (euclidean-based), embedded this position with all channel matrices in H. Run exhaustive search to find which RAs yield the best goodput for each channel matrix. Collect them. Repeat the step with another random position until we find the amount of data is sufficient for training. <br />

Dataset: Input: position (euclidean-based); Labels/Classes: RAs corresponding to that position. <br />

This dataset has a problem: Overlapping as shown in the picture.

![image](https://user-images.githubusercontent.com/15823161/52287627-d21e8b00-296a-11e9-9e8b-52aa4f4cb889.png)

Dissertation: 
Visualization of Raw dataset:
![ScPlotS1](https://user-images.githubusercontent.com/15823161/59569712-b43a0300-908d-11e9-83ae-09a3b7f6267d.jpg | width=50)

Processing directly to the raw dataset before feeding them through a neural network.
After being processed:

![ScPlotS1NN](https://user-images.githubusercontent.com/15823161/59569723-d6cc1c00-908d-11e9-9e8a-7a9478101cb6.jpg | width=50)

Overall system performance gained 10% without losing a single bit of raw data. 
# Built with 
Python 2.7 <br />
Matlab


