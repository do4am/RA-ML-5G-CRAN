function [V_lib, cl_size] = cl_simpleassignment2(iV, V_lib, i_neighbors, cl_size)
% Second phase: Unque Assignment
% @INPUT:
%iV index of current Vertice
%V library
%i_neighbors idex of neighbors to current vertex, descending order of distance
%current size of all classes

    n_idx = length(i_neighbors);
    class_name = {};

    Vi_classes_idx = find(V_lib(iV,:));
    Vi_classes = V_lib(iV, Vi_classes_idx); %all classes of the current vertice

    for i = 1:n_idx
        %export all the classes of the neighbors
        [~,~,class_name{i}] = find(V_lib(i_neighbors(i),:)); %#ok<AGROW>
    end 
        
    % find common classes between samples and its neighbors
    Vi_int_neighbors = intersect([class_name{:}], Vi_classes);  
    l = length(Vi_int_neighbors);
    
    % only one common_class
    if l == 1
        V_lib(iV,Vi_classes_idx) = 0;
        V_lib(iV,2) = Vi_int_neighbors;
        
        %update cl_size set  
        for i = 1:length(Vi_classes)
            cl_size(cl_size(:,1) == Vi_classes(i),2) = ...
                cl_size(cl_size(:,1) == Vi_classes(i),2) - 1;
        end 
            cl_size(cl_size(:,1) == V_lib(iV,2),2)  = ...
                cl_size(cl_size(:,1) == V_lib(iV,2),2)  + 1;
    
    % more than one common_class    
    elseif l > 1 %pick the largest sample class
        all_name = [class_name{:}];
        mode_int = zeros(1,l);
        for i = 1:l
            mode_int(i) = length(find(all_name == Vi_int_neighbors(i)));
        end
        
        n1 = find(mode_int == max(mode_int));

        if length(n1) == 1
            V_lib(iV,Vi_classes_idx) = 0;
            V_lib(iV,2) = Vi_int_neighbors(n1);
            
            %update cl_size set  
            for i = 1:length(Vi_classes)
                cl_size(cl_size(:,1) == Vi_classes(i),2) = ...
                    cl_size(cl_size(:,1) == Vi_classes(i),2) - 1;
            end 
                cl_size(cl_size(:,1) == V_lib(iV,2),2)  = ...
                    cl_size(cl_size(:,1) == V_lib(iV,2),2)  + 1;
            
        else
            max_cl_size = [0 0];
            for i = 1:l
                current_cl_size = cl_size(cl_size(:,1) == Vi_int_neighbors(i),2);
                if max_cl_size(1) < current_cl_size
                    max_cl_size = [current_cl_size i];
                end
            end 
            V_lib(iV,Vi_classes_idx) = 0;
            V_lib(iV,2) =  Vi_int_neighbors(max_cl_size(2));   
            
            %update cl_size set  
            for i = 1:length(Vi_classes)
                cl_size(cl_size(:,1) == Vi_classes(i),2) = ...
                    cl_size(cl_size(:,1) == Vi_classes(i),2) - 1;
            end 
                cl_size(cl_size(:,1) == V_lib(iV,2),2)  = ...
                    cl_size(cl_size(:,1) == V_lib(iV,2),2)  + 1;
        end
    else %Forcefully assignment l = 0
        [V_lib, cl_size] = cl_reassignment2(iV, V_lib, i_neighbors, cl_size);
    end 


