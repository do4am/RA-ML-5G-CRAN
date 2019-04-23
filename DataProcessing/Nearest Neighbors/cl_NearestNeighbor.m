function [V_lib,cl_size] = cl_NearestNeighbor(V, V_lib, cl_size, ClperV, K, mode)
% Choosing unique assignment to Sample based on neighbors.
% @INPUT:
%V is 2D matrix of points: pos vector (pos_x, pos_y)
%V_lib id class library, e.g., classes_library_updated
%cl_size: current size of all classes
%ClperV is number of class/RAs per point/vertex
%K: Number of nearest neighbors
%mode: First or Second phase. First phase by default.
% @OUTPUT:
% V_lib: updated library
% cl_Size: current size of all classes

    if nargin == 5
        mode = 1; 
    end
    
    %Sorting samples from top to bottome, from left to right
    [V_sorted, IV] = sortrows(V,[-3 2]); %ordered top-left
    [~,pre_IV] = sort(IV,'ascend');    
    V_lib = V_lib(IV,:);
    ClperV = ClperV(IV,:);
    V_cloud = pointCloud([V_sorted(:,2),V_sorted(:,3),zeros(length(V),1)]);
    
    for i = 1:size(V_sorted,1)   
        %considering common samples
        switch mode
            case 1
                if ClperV(i,1) > 1 %considering only samples belong to multiple classes
                    % find K neighbors
                    [indices,dist] = findNearestNeighbors(V_cloud,[V_sorted(i,2:3), 0], K+1);
                    % Since findNearestNeighbors also includes the consider sample, it should be removed
                    if length(dist(dist == 0)) > 1
                        indices(indices == i) = [];
                    else
                        indices(1) = [];
                    end
                    
                    %assignment if there is a single common class between sample and its neighbors
                    [V_lib, cl_size] = cl_simpleassignment(i, V_lib, indices, cl_size);
                end
                
            case 2               
                    [indices,dist] = findNearestNeighbors(V_cloud,[V_sorted(i,2:3), 0], K+1);
                    if length(dist(dist == 0)) > 1
                        indices(indices == i) = [];
                    else
                        indices(1) = [];
                    end

                   [V_lib, cl_size] = cl_simpleassignment2(i, V_lib, indices, cl_size);
        end
    end
    
    V_lib = V_lib(pre_IV,:);
    
    