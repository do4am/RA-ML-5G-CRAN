# Application of Machine learning to resource allocation for CRAN based 5G system
https://ieeexplore.ieee.org/document/8369028 

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

One note is, the exhaustive search is actually the way the station perform after receiving the CSI estimated from the receiver to determine which RA should be assigned to that UE. Therefore, in term of goodput performance, the CSI-based approach alway yields the best result. The idea is to try achive as close as goodput performance of CSI-based approach as possible utilizing the learning-based approach. <br />

Dataset: Input: position (euclidean-based); Labels/Classes: RAs corresponding to that position. <br />

This dataset has a problem: One input data belongs to different classes (figure below). Datapoints line within the overlapping region will cause the confusion in the training process (for Random Forest and deep neural network). And thus the training results are not optimised. <br />

![image](https://user-images.githubusercontent.com/15823161/52287627-d21e8b00-296a-11e9-9e8b-52aa4f4cb889.png)

Need a criteria to remove the overlapping, so that, one input data belongs to one and only one class. This process of removing overlapping called homogeneousity - removal of overlapping to maximise the learning process - we expect that the goodput performance will be closer to the that of CSI-based approach <br />

A heuristic solution for homogenousity is applied first and we already see some improvement. An optimal solution is still on study.

# Built with 
Python 2.7 <br />
Matlab


