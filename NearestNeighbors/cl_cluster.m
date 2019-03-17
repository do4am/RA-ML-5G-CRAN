function [V_lib,cl_size] = cl_cluster(V, V_lib, cl_size, ClperV, K, mode)
%V is 2D matrix of points 
%V library
%current size of all classes
%ClperV is number of class/RAs per point/vertice
%Number of nearest neighbors
    if nargin == 5
        mode = 1;
    end

    [V_sorted, IV] = sortrows(V,[-3 2]); %ordered top-left
    [~,pre_IV] = sort(IV,'ascend');    
    V_lib = V_lib(IV,:);
    ClperV = ClperV(IV,:);
    V_cloud = pointCloud([V_sorted(:,2),V_sorted(:,3),zeros(length(V),1)]);
    
    for i = 1:size(V_sorted,1)   
        %considering common samples
        switch mode
            case 1
                if ClperV(i,1) > 1 %considering only samples belong to multi-classes
                    [indices,dist] = findNearestNeighbors(V_cloud,[V_sorted(i,2:3), 0], K+1);
                    %remove neighbor of itself
                    if length(dist(dist == 0)) > 1
                        indices(indices == i) = [];
                    else
                        indices(1) = [];
                    end
                    
                    %dists(1) = [];

                    %update V_library after each assignment
                    [V_lib, cl_size] = cl_assignment(i, V_lib, indices, cl_size);
                end
                
            case 2               
                if ClperV(i,1) == 1 %considering only samples belong to 1 classes
                    [indices,dist] = findNearestNeighbors(V_cloud,[V_sorted(i,2:3), 0], K+1);
                    %remove neighbor of itself
                    if length(dist(dist == 0)) > 1
                        indices(indices == i) = [];
                    else
                        indices(1) = [];
                    end
                    %dists(1) = [];

                    %update V_library after each assignment
                    [V_lib, cl_size] = cl_assignment(i, V_lib, indices, cl_size);
                end 
                
            case 3
                if ClperV(i,1) >= 1 %considering all
                    [indices,dist] = findNearestNeighbors(V_cloud,[V_sorted(i,2:3), 0], K+1);
                    %remove neighbor of itself
                    if length(dist(dist == 0)) > 1
                        indices(indices == i) = [];
                    else
                        indices(1) = [];
                    end
                    %dists(1) = [];

                    %update V_library after each assignment
                    [V_lib, cl_size] = cl_assignment(i, V_lib, indices, cl_size);
                end 
        end
    end
    
    V_lib = V_lib(pre_IV,:);
    
    